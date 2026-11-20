"""Write a synthetic corpus of retail support tickets, with fake personal details, to corpus/raw_interactions.txt.

Every name, email, phone number, order number, card number and address here is invented. They are put in on
purpose so that the anonymizer (Milestone 1 in app/core.py) has something real-looking to remove.

    python scripts/generate_raw_corpus.py              # 600 tickets, seed 7
    python scripts/generate_raw_corpus.py --tickets 1000 --seed 1
"""
import argparse
import random
from pathlib import Path

FIRST = ["Sarah", "James", "Priya", "Miguel", "Aisha", "Tom", "Mei", "Daniel", "Fatima", "Lucas", "Olivia", "Ravi",
         "Emma", "Noah", "Chloe", "Arjun", "Hannah", "Diego", "Grace", "Omar", "Sofia", "Ben", "Leah", "Kenji"]
LAST = ["Johnson", "Smith", "Patel", "Garcia", "Khan", "Nguyen", "Brown", "Rossi", "Okafor", "Müller", "Silva",
        "Kim", "Taylor", "Lee", "Martin", "Rao", "Wilson", "Lopez", "Ahmed", "Clarke"]
AGENTS = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "example.org", "mail.com"]
STREETS = ["Maple Street", "Oak Avenue", "Pine Road", "Elm Drive", "Cedar Lane", "Park Boulevard", "Lake View Road"]
CITIES = ["Springfield", "Riverton", "Fairview", "Lakewood", "Greenville", "Bristol"]
PRODUCTS = ["running shoes", "winter jacket", "coffee maker", "bluetooth headphones", "yoga mat", "desk lamp",
            "backpack", "blender", "smart watch", "bed sheets", "phone case", "kettle", "jeans", "sneakers",
            "air fryer", "water bottle", "wireless mouse", "sofa cushion", "dinner set", "hair dryer"]
CARRIERS = ["UPS", "FedEx", "DHL", "USPS"]


class Person:
    def __init__(self, rng: random.Random):
        self.first, self.last = rng.choice(FIRST), rng.choice(LAST)
        sep = rng.choice([".", "_", ""])
        self.email = f"{self.first.lower()}{sep}{self.last.lower()}{rng.randint(1, 99)}@{rng.choice(DOMAINS)}"
        self.phone = rng.choice([
            f"({rng.randint(200, 989)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            f"{rng.randint(200, 989)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            f"+1 {rng.randint(200, 989)} {rng.randint(200, 999)} {rng.randint(1000, 9999)}",
        ])
        self.address = f"{rng.randint(10, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)} {rng.randint(10000, 99999)}"


def order_id(rng):
    return rng.choice([f"#{rng.randint(100000, 999999)}", f"ORD-{rng.randint(100000, 999999)}",
                       str(rng.randint(1000000, 9999999))])


def card(rng):
    return " ".join(str(rng.randint(1000, 9999)) for _ in range(4))


def tracking(rng):
    return "1Z" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(16))


# --- what customers write (slots: {name} {email} {phone} {order} {product} {card} {address} {days})
CUSTOMER = {
    "where_is_order": [
        "Hi, my name is {name}. I ordered a {product} last week, {order}, and it still hasn't arrived. Where is it?",
        "Hello, I placed {order} {days} days ago and the tracking hasn't updated. My email is {email}.",
        "Hi there, can you tell me when my {product} will be delivered? Order {order}. Thanks, {name}",
        "My package is late. The order is {order} and I really need the {product} by Friday. Call me at {phone}.",
    ],
    "refund": [
        "Hi, I returned the {product} two weeks ago but I haven't received my refund yet. Order {order}.",
        "Hello, I'm {name}. I was charged twice for {order} on my card {card}. Please refund the extra charge.",
        "When will I get my money back for the {product}? I sent it back {days} days ago. Email: {email}",
    ],
    "return": [
        "Hi, I'd like to return the {product} from {order}. It doesn't fit. How do I do that?",
        "Hello, how can I return an item? I bought a {product} and changed my mind. This is {name}, {email}.",
        "Can I exchange the {product} for a different size? Order {order}.",
    ],
    "damaged": [
        "Hi, my {product} arrived broken. The box was crushed. Order {order}. What can you do?",
        "Hello, the {product} I received is damaged and doesn't work. My name is {name}, phone {phone}.",
        "The {product} came with a big scratch on it. I want a replacement. Order {order}.",
    ],
    "wrong_item": [
        "I received the wrong item. I ordered a {product} but got something else. Order {order}.",
        "Hi, this is {name}. My order {order} was missing the {product}. Only part of it arrived.",
    ],
    "cancel": [
        "Please cancel my order {order}. I ordered the {product} by mistake.",
        "Hi, can I still cancel {order}? I found the {product} cheaper elsewhere. Thanks, {name}",
    ],
    "address": [
        "Hi, I need to change the delivery address for {order}. The new address is {address}.",
        "I moved last week. Can you send my {product} to {address} instead? My email is {email}.",
    ],
    "payment": [
        "My card {card} was declined at checkout but I have enough money. Can you help?",
        "Hi, I can't pay for my order. The payment page keeps showing an error. This is {name}, {email}.",
    ],
    "account": [
        "I can't log in to my account. I reset my password but never got the email. My email is {email}.",
        "Hello, please update the phone number on my account to {phone}. My name is {name}.",
    ],
    "promo": [
        "Hi, my discount code isn't working at checkout. I'm trying to buy a {product}.",
        "I forgot to apply my coupon to {order}. Can you still give me the discount? Thanks, {name}",
    ],
    "stock": [
        "Hi, when will the {product} be back in stock? I'd like to buy one.",
        "Is the {product} available in a larger size? My name is {name}.",
    ],
}

# --- what agents write back. Built from shared openers and closers plus intent-specific middles,
# so the model sees the same useful phrases many times with some variety.
OPENERS = [
    "Hi {name}, thank you for contacting us.",
    "Hello {name}, thank you for reaching out.",
    "Hi {name}, thanks for getting in touch.",
    "Dear {name}, thank you for contacting our support team.",
    "Hi {name}, thank you for your message.",
    "Hello {name}, thanks for reaching out to us.",
]
APOLOGIES = [
    "I'm sorry for the inconvenience.",
    "I'm really sorry about this.",
    "I apologize for the trouble this has caused.",
    "I'm sorry to hear about this.",
    "We apologize for the inconvenience.",
]
CLOSERS = [
    "If you have any other questions, please let us know.",
    "Please let us know if there is anything else we can help you with.",
    "If you need anything else, feel free to reply to this email.",
    "Thank you for your patience.",
    "Thank you for shopping with us.",
    "Is there anything else I can help you with today?",
]
SIGNOFFS = ["Best regards, {agent}", "Kind regards, {agent}", "Thanks, {agent}", "Best, {agent} from Customer Support",
            "Have a great day! {agent}"]

AGENT = {
    "where_is_order": [
        "I checked your order {order} and it was shipped with {carrier} on Monday. Your tracking number is {tracking}.",
        "I can see that your order {order} is on its way and should arrive within {days} business days.",
        "Your order {order} was delayed at the warehouse, but it has now been shipped. You will receive a tracking email shortly.",
        "I have contacted {carrier} about your package and they expect to deliver it within {days} business days.",
        "According to our records, your {product} is out for delivery today.",
    ],
    "refund": [
        "I can confirm that we received your return and your refund has been processed. It may take {days} business days to appear on your statement.",
        "I have issued a full refund to your original payment method. You should see it within {days} business days.",
        "I can see the duplicate charge on your order {order}. I have refunded the extra payment to your card ending in {last4}.",
        "Your refund is being processed and you will receive a confirmation email at {email} once it is complete.",
    ],
    "return": [
        "You can return the {product} within 30 days of delivery. I have sent a prepaid return label to {email}.",
        "To start a return, please log in to your account, go to your orders, and select the item you want to return.",
        "We would be happy to exchange the {product} for a different size. I have sent you a return label by email.",
        "Please pack the item in its original packaging and drop it off at any {carrier} location. Once we receive it, we will process your refund.",
    ],
    "damaged": [
        "I'm sorry that your {product} arrived damaged. We will send you a replacement at no extra cost.",
        "Could you please send us a photo of the damaged {product}? Once we receive it, we will send you a replacement or a full refund.",
        "I have arranged a free replacement for your {product}. You do not need to return the damaged item.",
        "We will send you a replacement right away, and you will receive a tracking number by email.",
    ],
    "wrong_item": [
        "I'm sorry that you received the wrong item. We will send the correct {product} right away, and I have sent a prepaid return label to {email}.",
        "I can see that the {product} was missing from your order {order}. We will ship it to you today at no extra cost.",
    ],
    "cancel": [
        "I have cancelled your order {order}, and you will receive a full refund within {days} business days.",
        "Unfortunately, your order {order} has already been shipped, so we cannot cancel it. You can return it for a full refund once it arrives.",
        "Your order {order} has been cancelled. You will receive a confirmation email shortly.",
    ],
    "address": [
        "I have updated the delivery address for your order {order}. Your package will be delivered to the new address.",
        "Unfortunately, your order {order} has already been shipped, so we cannot change the address. Please contact {carrier} with your tracking number {tracking}.",
    ],
    "payment": [
        "Please check that the billing address matches the address on your card, and then try again.",
        "I can see that the payment was declined by your bank. Please contact your bank, or try a different payment method.",
        "We are aware of a problem with our payment page and our team is working on it. Please try again in a few hours.",
    ],
    "account": [
        "I have sent a new password reset link to {email}. Please check your spam folder if you do not see it.",
        "I have updated the phone number on your account. You can check your account details at any time on our website.",
        "Your account has been unlocked. You should now be able to log in with your email and password.",
    ],
    "promo": [
        "I have applied the discount to your order {order}, and the difference will be refunded to your card.",
        "Please make sure the code is entered exactly as shown and that it has not expired. Discount codes cannot be combined with other offers.",
        "I'm sorry, but this discount code has expired. As a thank you for your patience, I have added a 10% coupon to your account.",
    ],
    "stock": [
        "The {product} is currently out of stock, but we expect it to be back in stock within {days} weeks.",
        "You can sign up on the product page to receive an email when the {product} is back in stock.",
        "Yes, the {product} is available in a larger size. You can find all available sizes on the product page.",
    ],
}
NEEDS_APOLOGY = {"where_is_order", "refund", "damaged", "wrong_item", "payment", "promo"}


def ticket(rng: random.Random, number: int) -> str:
    intent = rng.choice(sorted(CUSTOMER))
    person = Person(rng)
    slots = {
        "name": person.first if rng.random() < 0.6 else f"{person.first} {person.last}",
        "email": person.email, "phone": person.phone, "address": person.address,
        "order": order_id(rng), "product": rng.choice(PRODUCTS), "card": card(rng),
        "days": rng.choice([2, 3, 4, 5, 7, 10]), "carrier": rng.choice(CARRIERS), "tracking": tracking(rng),
        "agent": rng.choice(AGENTS),
    }
    slots["last4"] = slots["card"][-4:]
    customer = rng.choice(CUSTOMER[intent]).format(**slots)

    reply = [rng.choice(OPENERS).format(name=person.first)]
    if intent in NEEDS_APOLOGY and rng.random() < 0.8:
        reply.append(rng.choice(APOLOGIES))
    middles = AGENT[intent]
    for middle in rng.sample(middles, k=min(len(middles), rng.choice([1, 1, 2]))):
        reply.append(middle.format(**slots))
    reply.append(rng.choice(CLOSERS))
    reply.append(rng.choice(SIGNOFFS).format(agent=slots["agent"]))
    if rng.random() < 0.15:  # agents sometimes leave a direct line, which must be anonymized too
        reply.insert(-1, f"You can also call me directly at {Person(rng).phone}.")
    return f"### Ticket {number} | {intent}\nCustomer: {customer}\nAgent: {' '.join(reply)}\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickets", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="corpus/raw_interactions.txt")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ticket(rng, 10_000 + i) for i in range(args.tickets)), encoding="utf-8")
    print(f"wrote {args.tickets} tickets to {out}")


if __name__ == "__main__":
    main()
