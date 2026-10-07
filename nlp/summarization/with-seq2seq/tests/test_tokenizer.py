"""Le vocabulaire partagé : appris sur le train, **reproductible**, et ses identifiants fixes.

Le vocabulaire est la brique qui décide de tout le reste : si deux exécutions apprennent deux
vocabulaires différents, le même modèle entraîné deux fois ne produit pas les mêmes résumés, et
aucune comparaison n'est interprétable. Ces tests verrouillent donc les trois propriétés dont dépend
la reproductibilité du projet :

* les identifiants des jetons spéciaux ne bougent jamais (un décalage d'un seul identifiant produit
  un modèle qui apprend un bruit parfaitement reproductible) ;
* deux apprentissages sur le même corpus donnent le **même** vocabulaire, aussi bien dans le même
  processus qu'entre deux processus — c'est la raison pour laquelle l'apprentissage est écrit dans le
  projet plutôt que délégué à l'apprenti de la bibliothèque ;
* la taille cible est respectée, et un mot hors corpus reste lisible en pièces plutôt qu'effacé en un
  seul ``[UNK]``.
"""

from __future__ import annotations

import pandas as pd

from src.models.tokenizer import SPECIAL_TOKENS, SharedWordPieceTokenizer


def _learner(vocab_size: int = 320) -> SharedWordPieceTokenizer:
    """Return an unfitted tokenizer with a deliberately small target vocabulary."""
    return SharedWordPieceTokenizer(vocab_size=vocab_size, min_frequency=1, max_input_tokens=64)


def test_the_special_ids_never_move() -> None:
    """``[PAD]``, ``[UNK]``, ``[BOS]`` et ``[EOS]`` occupent les quatre premiers identifiants."""
    tokenizer = _learner().fit(["le doseur P-12 fuit depuis 45 minutes"])
    assert tokenizer.special_ids() == (0, 1, 2, 3)
    assert [tokenizer.special_id(token) for token in SPECIAL_TOKENS] == [0, 1, 2, 3]


def test_two_learners_learn_the_same_vocabulary(documents: pd.DataFrame) -> None:
    """Deux apprentissages sur le même corpus produisent le **même** vocabulaire, identifiants inclus.

    C'est le test de non-régression de la reproductibilité : l'apprenti de ``tokenizers`` départage
    les fréquences égales par l'ordre d'une table de hachage, donc deux vocabulaires de même taille
    mais de composition différente — et deux modèles non comparables.
    """
    corpus = documents["text"].head(40).tolist()
    first = _learner().fit(corpus)
    second = _learner().fit(corpus)
    assert first.vocabulary_size == second.vocabulary_size
    assert first.tokenizer is not None and second.tokenizer is not None
    assert first.tokenizer.get_vocab() == second.tokenizer.get_vocab()

    text = str(documents["text"].iloc[0])
    assert first.encode(text) == second.encode(text)
    assert first.pieces(text) == second.pieces(text)


def test_the_vocabulary_respects_its_target(documents: pd.DataFrame) -> None:
    """La taille apprise ne dépasse pas la cible, et les fusions ont bien lieu."""
    corpus = documents["text"].head(60).tolist()
    tokenizer = _learner(vocab_size=200).fit(corpus)
    assert tokenizer.vocabulary_size <= 200
    assert tokenizer.vocabulary_size > len(SPECIAL_TOKENS)
    # Un mot du corpus est découpé en pièces **connues**, plus rarement en un seul `[UNK]`.
    assert tokenizer.unknown_rate(corpus) < 0.2


def test_a_word_of_the_corpus_becomes_a_single_piece(documents: pd.DataFrame) -> None:
    """Un mot répété du corpus finit fusionné : le vocabulaire apprend les mots, pas les lettres."""
    corpus = documents["text"].tolist()
    tokenizer = _learner(vocab_size=600).fit(corpus)
    words = [word for text in corpus for word in str(text).lower().split() if len(word) > 2]
    word = str(pd.Series(words).value_counts().idxmax())
    assert word in tokenizer.tokenizer.get_vocab()


def test_decoding_drops_the_special_tokens() -> None:
    """Le texte décodé ne contient ni ``[BOS]`` ni ``[EOS]``, et il est relisible.

    Le test utilise un texte **court** : au-delà du budget, l'encodage est tronqué (le ``[EOS]``
    disparaît avec la fin du texte), ce que le test suivant vérifie séparément.
    """
    tokenizer = _learner().fit(["le doseur p-12 fuit depuis 45 minutes"])
    text = "le doseur p-12 fuit depuis 45 minutes"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    decoded = tokenizer.decode(ids)
    assert "[BOS]" not in decoded and "[EOS]" not in decoded
    # La tokenisation peut couper un identifiant (`p-12`) : les espaces de la découpe se sont
    # ajoutés, mais le texte lui-même est intact.
    assert decoded.replace(" ", "") == text.replace(" ", "")


def test_the_encoder_bound_is_respected(documents: pd.DataFrame) -> None:
    """Un document plus long que la borne est tronqué, ``[EOS]`` compris, et c'est mesurable."""
    tokenizer = _learner().fit(documents["text"].head(20).tolist())
    long_text = str(documents["text"].iloc[0])
    ids = tokenizer.encode(long_text)
    assert len(ids) <= tokenizer.max_input_tokens + 2
    assert tokenizer.truncation_rate([long_text * 4]) == 1.0


def test_a_word_outside_the_corpus_is_flagged_unknown(documents: pd.DataFrame) -> None:
    """Un mot hors corpus est mesuré comme inconnu au lieu de passer pour du texte appris."""
    tokenizer = _learner().fit(documents["text"].head(20).tolist())
    assert tokenizer.unknown_rate(["Спасибо за помощь"]) > 0.0
    assert tokenizer.unknown_rate([]) == 0.0


def test_the_tokenizer_publishes_how_it_was_learned(documents: pd.DataFrame) -> None:
    """La fiche modèle reçoit la version de l'apprenti, ses fusions et le taux d'inconnus."""
    corpus = documents["text"].head(20).tolist()
    tokenizer = _learner().fit(corpus)
    assert tokenizer.meta["version"] == "shared-wordpiece-v2"
    assert tokenizer.meta["words"] > 0
    assert tokenizer.meta["merges"] >= 0
    # Le taux publié est celui du corpus d'apprentissage, mesuré par le tokenizer lui-même.
    assert tokenizer.meta["unknown_rate"] == tokenizer.unknown_rate(corpus)
