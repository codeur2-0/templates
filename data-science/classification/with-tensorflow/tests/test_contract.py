from loan_tensorflow.data import LoanDataLoader


def test_credit_scores_are_in_valid_range():
    frame = LoanDataLoader("data/raw/loans.csv").load()
    assert frame.credit_score.between(300, 850).all()
