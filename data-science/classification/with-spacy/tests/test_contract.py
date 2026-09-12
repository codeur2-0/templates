from ticket_spacy.data import TicketDataLoader


def test_ticket_labels_are_known():
    frame = TicketDataLoader("data/raw/tickets.csv").load()
    assert set(frame.label) == {"billing", "technical", "account"}
