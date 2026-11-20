# Sample suggestions

Model: n=4, Kneser-Ney smoothing. Produced by `python -m app.cli report`.

## (empty: start of a response)

Next words: `hi` 50%, `hello` 34%, `dear` 16%

1. Hi <NAME>, thanks for getting in touch.
2. Hello <NAME>, thanks for getting in touch.
3. Hi <NAME>, thank you for your patience.

## "Hi Sarah, thank you for"

Next words: `your` 36%, `contacting` 31%, `reaching` 17%, `shopping` 16%

1. Hi Sarah, thank you for contacting our support team.
2. Hi Sarah, thank you for shopping with us.
3. Hi Sarah, thank you for your patience.

## "I'm sorry"

Next words: `to` 12%, `for` 11%, `,` 11%, `that` 10%, `about` 4%

1. I'm sorry, but this discount code has expired.
2. I'm sorry to hear about this.
3. I'm sorry for the inconvenience.

## "I have issued a full"

Next words: `refund` 99%

1. I have issued a full refund once it arrives.
2. I have issued a full refund.
3. I have issued a full refund to your original payment method.

## "Your order"

Next words: `<ORDER_ID>` 94%

1. Your order <ORDER_ID> has already been shipped, so we cannot change the address.
2. Your order <ORDER_ID>.
3. Your order <ORDER_ID> has already been shipped, so we cannot cancel it.

## "We will send"

Next words: `you` 61%, `the` 38%

1. We will send you a replacement at no extra cost.
2. We will send you a replacement right away, and I have sent a prepaid return label to <EMAIL>.
3. We will send you a replacement or a full refund once it arrives.

## "Please send us a photo"

Next words: `of` 98%

1. Please send us a photo of the damaged water bottle?
2. Please send us a photo of the damaged hair dryer?
3. Please send us a photo of the damaged dinner set?

## "To start a return,"

Next words: `please` 97%

1. To start a return, please log in to your account.
2. To start a return, please log in to your account, go to your orders, and select the item.
3. To start a return, please let us know if there is anything else we can help you with today?
