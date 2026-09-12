from fraud_pytorch.data import TransactionDataLoader


def test_transactions_are_validated():
    frame = TransactionDataLoader("data/raw/transactions.csv").load()
    assert frame["country_risk"].between(0, 1).all()
    assert len(frame) == 16
