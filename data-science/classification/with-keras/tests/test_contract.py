from sentiment_keras.data import ReviewDataLoader


def test_review_labels_are_constrained():
    frame = ReviewDataLoader("data/raw/reviews.csv").load()
    assert set(frame.sentiment) <= {"positive", "neutral", "negative"}
