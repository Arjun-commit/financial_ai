from fin_flow.utils.pii import mask_pii


def test_mask_email():
    assert mask_pii("payment to john.doe@example.com") == "payment to [EMAIL]"


def test_mask_ssn():
    assert mask_pii("ssn 123-45-6789 on file") == "ssn [SSN] on file"


def test_mask_card_number():
    out = mask_pii("card 4111 1111 1111 1111 charged")
    assert "[CARD]" in out
    assert "4111" not in out


def test_mask_long_account_digits():
    assert mask_pii("acct 123456789 wire") == "acct [ACCT] wire"


def test_short_numbers_preserved():
    # Amounts and small numbers shouldn't be masked
    assert mask_pii("coffee $5.75 at store #42") == "coffee $5.75 at store #42"
