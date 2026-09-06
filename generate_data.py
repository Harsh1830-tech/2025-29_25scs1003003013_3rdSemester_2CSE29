"""
Generates data/emails.csv — a labeled spam/ham email dataset.

There's no bundled internet-sourced dataset here (e.g. Enron/SpamAssassin)
to keep the repo dependency-free and license-clean. Instead this script
procedurally builds realistic emails from templates + word banks covering
common spam patterns (prizes, urgency, phishing, pharma, crypto/loans) and
common everyday ham patterns (work, scheduling, receipts, personal notes).

Run:
    python src/generate_data.py
Writes data/emails.csv with columns: label ('spam'/'ham'), text
"""

import csv
import random

random.seed(42)

# ---------------------------------------------------------------------
# SPAM building blocks
# ---------------------------------------------------------------------
spam_openers = [
    "CONGRATULATIONS!!!", "URGENT NOTICE", "Dear Winner", "ATTENTION",
    "FINAL REMINDER", "Hello Valued Customer", "Dear Beneficiary",
    "GOOD NEWS", "ACT NOW", "Important Update", "Hi there!!!",
    "Limited Time Offer", "You have been selected", "Breaking:",
]

spam_bodies = [
    "You have WON ${amt} in the {lottery} Lottery! To claim your prize, click the link below and enter your bank details within 24 hours.",
    "Your account will be SUSPENDED unless you verify your password immediately. Click here to confirm your identity: {link}",
    "Make ${amt} per week working from home! No experience needed. Just send a small registration fee of $49 to get started TODAY.",
    "Hot singles in your area are waiting to chat with YOU right now! Click below for FREE access, no credit card required!!!",
    "Buy cheap {pharma} online with 80% discount, no prescription needed! Fast, discreet shipping worldwide. Order now!",
    "Your package could not be delivered. Pay a redelivery fee of $2.99 immediately or your parcel will be returned: {link}",
    "This is your FINAL WARNING. Your {service} subscription has expired. Update your billing information now to avoid account closure.",
    "Invest in {crypto} today and turn $100 into $10,000 in just 7 days!! Guaranteed returns, act before this offer closes!",
    "We noticed unusual sign-in activity on your account. Verify your identity now or your account will be permanently locked: {link}",
    "Get a pre-approved loan of ${amt} with ZERO credit check! Apply now, funds sent within 1 hour, no paperwork required.",
    "CLICK HERE to claim your free iPhone {model}! Limited stock available, only 3 left, hurry before it's too late!!!",
    "Lose {kg}kg in just 2 weeks with this ONE weird trick doctors don't want you to know about! Order your free trial now.",
    "As the sole heir, you are entitled to a inheritance of ${amt} from a distant relative. Reply with your bank details to proceed.",
    "Your computer has a VIRUS! Download our antivirus tool immediately to remove 27 threats detected on your device: {link}",
    "Refinance your mortgage today and SAVE thousands! Rates as low as 0.9% APR, approval guaranteed regardless of credit score.",
    "You've been chosen for a FREE cruise to the Bahamas! Just pay a small processing fee of $99 to reserve your spot now.",
    "Your {service} password was reset. If this wasn't you, click below immediately to secure your account: {link}",
    "Earn passive income of ${amt}/month with our proven system. No skills, no experience, just click below to start earning today!",
    "URGENT: Your tax refund of ${amt} is pending. Submit your social security number and banking info to release the funds.",
    "Meet gorgeous women near you tonight! Sign up FREE, no strings attached, thousands of members online right now!!!",
]

spam_closers = [
    "Click here now: {link}", "Reply IMMEDIATELY to claim.", "Offer expires in 24 hours, act fast!",
    "This is not a scam, 100% legit and guaranteed!!!", "Limited spots available, don't miss out!",
    "For your security, verify now: {link}", "Don't delay, click below to proceed.",
    "100% risk free, cancel anytime.", "Text STOP to unsubscribe (but hurry, offer ends soon)!",
]

lotteries = ["International", "Euro Millions", "Global Sweepstakes", "Mega Millions", "Coca-Cola Promo"]
pharmas = ["Viagra", "Xanax", "weight-loss pills", "prescription meds", "painkillers"]
services = ["Netflix", "PayPal", "Amazon Prime", "Apple ID", "bank", "email"]
cryptos = ["Bitcoin", "Ethereum", "a new crypto token", "Dogecoin"]
links = ["http://claim-your-prize.example.ru/verify", "http://secure-login-update.example.info",
         "http://bit.ly/free-cash-now", "http://account-verify.example.top"]

# ---------------------------------------------------------------------
# HAM (legitimate) building blocks
# ---------------------------------------------------------------------
ham_openers = [
    "Hi Team,", "Hey,", "Hi John,", "Good morning,", "Hello,", "Hi all,",
    "Hi Sarah,", "Dear Priya,", "Hey team,", "Hi,",
]

ham_bodies = [
    "Just a reminder that our {meeting} is scheduled for {day} at {time}. Please let me know if that time doesn't work for you.",
    "I've attached the {doc} we discussed yesterday. Let me know if you have any questions or need changes before Friday.",
    "Thanks for sending over the report. I reviewed it and left a few comments in the shared doc. Overall looks great!",
    "Can we push our {meeting} to {day}? Something came up on my end and I want to make sure I'm fully prepared.",
    "Here's the invoice for {item} — total comes to ${amt2}. Let me know if you need it in a different format.",
    "Happy birthday! Hope you have a wonderful day, let's catch up for coffee sometime next week if you're free.",
    "The kids' school project is due on {day}, can you pick up some poster board on your way home tonight?",
    "Following up on our call from earlier — I'll send the updated {doc} by end of day tomorrow.",
    "Just confirming our reservation at {place} for {day} at {time}. Looking forward to seeing everyone!",
    "Quick question about the {item} order — did it ship yet? Tracking still shows it's processing.",
    "Great meeting you at the conference! Would love to grab lunch and talk more about the {item} project sometime.",
    "The quarterly numbers are in — revenue is up 8% from last quarter, details are in the attached spreadsheet.",
    "Reminder: your dentist appointment is on {day} at {time}. Call the office if you need to reschedule.",
    "Thanks so much for your help moving last weekend, really appreciate it! Let's grab dinner soon to say thanks properly.",
    "Could you review the pull request I opened this morning? Nothing urgent, just want your eyes on the {item} changes.",
    "Here are my notes from today's {meeting} — action items are highlighted at the bottom, due by {day}.",
    "Your order of {item} has shipped and should arrive by {day}. You can track it using the link in your account.",
    "Loved catching up with you last night! We should plan that trip to {place} soon, maybe over the long weekend.",
    "Attaching the {doc} for tomorrow's {meeting}. Let me know if I missed anything before I send it to the client.",
    "Can you send over your availability for next week? Want to lock in time for the {item} kickoff.",
]

meetings = ["team sync", "1:1", "project review", "client call", "standup", "budget meeting"]
docs = ["proposal", "slide deck", "contract", "meeting notes", "design mockups", "spreadsheet"]
items = ["laptop", "office chairs", "marketing", "onboarding", "Q3 roadmap", "website redesign"]
places = ["that new Italian place", "the downtown office", "the lake house", "Marco's"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "next week"]
times = ["10:00 AM", "2:30 PM", "9:00 AM", "4:00 PM", "11:30 AM"]


def fill(template):
    return template.format(
        amt=random.choice([500, 1000, 5000, 10000, 25000, 250000, 1000000]),
        amt2=random.choice([120, 349.99, 899, 45.50, 1200]),
        lottery=random.choice(lotteries),
        pharma=random.choice(pharmas),
        service=random.choice(services),
        crypto=random.choice(cryptos),
        link=random.choice(links),
        model=random.choice(["15", "15 Pro", "14", "SE"]),
        kg=random.choice([5, 8, 10, 15]),
        meeting=random.choice(meetings),
        doc=random.choice(docs),
        item=random.choice(items),
        place=random.choice(places),
        day=random.choice(days),
        time=random.choice(times),
    )


def make_spam():
    parts = [random.choice(spam_openers), fill(random.choice(spam_bodies))]
    if random.random() < 0.8:
        parts.append(fill(random.choice(spam_closers)))
    text = " ".join(parts)
    # Spam often has erratic punctuation/caps - nudge a little more sometimes.
    if random.random() < 0.3:
        text = text.replace(".", "!")
    return text


def make_ham():
    parts = [random.choice(ham_openers), fill(random.choice(ham_bodies))]
    if random.random() < 0.5:
        parts.append(random.choice(["Thanks!", "Best,", "Talk soon.", "Appreciate it.", "Let me know either way."]))
    return " ".join(parts)


def main():
    rows = []
    n_per_class = 400
    for _ in range(n_per_class):
        rows.append(("spam", make_spam()))
        rows.append(("ham", make_ham()))

    random.shuffle(rows)

    # De-duplicate exact-text collisions from limited template combinations
    seen = set()
    unique_rows = []
    for label, text in rows:
        if text not in seen:
            seen.add(text)
            unique_rows.append((label, text))

    with open("data/emails.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(unique_rows)

    n_spam = sum(1 for l, _ in unique_rows if l == "spam")
    n_ham = sum(1 for l, _ in unique_rows if l == "ham")
    print(f"Wrote {len(unique_rows)} emails to data/emails.csv ({n_spam} spam, {n_ham} ham)")


if __name__ == "__main__":
    main()
