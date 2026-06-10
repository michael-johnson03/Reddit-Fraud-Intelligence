# fraud_reddit_sentiment/inference/fraud_taxonomy.py
"""Fraud taxonomy: high-confidence phrases and theme word bags.

Two-tier classification:
- THEME_KEYWORDS: exact phrase matches (high confidence, higher weight)
- THEME_WORD_BAGS: individual word hits (broader coverage, lower weight per hit)

Used by theme_classify.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier 1 — Exact phrase matches
# Each entry: (phrase, theme, weight)
# ---------------------------------------------------------------------------

THEME_KEYWORDS = [

    # --- ATO: Account Takeover ---
    ("account takeover", "ATO_Account_Takeover", 1.5),
    ("account takeovers", "ATO_Account_Takeover", 1.5),
    ("ato", "ATO_Account_Takeover", 1.2),
    ("hacked account", "ATO_Account_Takeover", 1.3),
    ("account hacked", "ATO_Account_Takeover", 1.3),
    ("account compromised", "ATO_Account_Takeover", 1.3),
    ("compromised account", "ATO_Account_Takeover", 1.3),
    ("account recovery", "ATO_Account_Takeover", 1.2),
    ("account disabled", "ATO_Account_Takeover", 1.2),
    ("account locked", "ATO_Account_Takeover", 1.2),
    ("unauthorized login", "ATO_Account_Takeover", 1.4),
    ("unauthorized access", "ATO_Account_Takeover", 1.3),
    ("password reset", "ATO_Account_Takeover", 1.2),
    ("password changed", "ATO_Account_Takeover", 1.2),
    ("locked out", "ATO_Account_Takeover", 1.1),
    ("two factor", "ATO_Account_Takeover", 1.0),
    ("two-factor", "ATO_Account_Takeover", 1.0),
    ("2fa", "ATO_Account_Takeover", 1.1),
    ("mfa", "ATO_Account_Takeover", 1.1),
    ("sim swap", "ATO_Account_Takeover", 1.4),
    ("sim-swap", "ATO_Account_Takeover", 1.4),
    ("recovery email", "ATO_Account_Takeover", 1.2),
    ("recovery code", "ATO_Account_Takeover", 1.2),
    ("verification code", "ATO_Account_Takeover", 1.2),
    ("login attempt", "ATO_Account_Takeover", 1.2),
    ("credential stuffing", "ATO_Account_Takeover", 1.4),
    ("credential theft", "ATO_Account_Takeover", 1.4),
    ("someone logged into", "ATO_Account_Takeover", 1.4),
    ("someone accessed", "ATO_Account_Takeover", 1.3),
    ("account stolen", "ATO_Account_Takeover", 1.3),
    ("account breached", "ATO_Account_Takeover", 1.3),
    ("logged in without", "ATO_Account_Takeover", 1.3),
    ("someone hacked", "ATO_Account_Takeover", 1.3),
    ("email hacked", "ATO_Account_Takeover", 1.3),
    ("phone hacked", "ATO_Account_Takeover", 1.3),

    # --- Phishing / Smishing / Vishing ---
    ("phishing", "Phishing_Smishing", 1.4),
    ("smishing", "Phishing_Smishing", 1.4),
    ("vishing", "Phishing_Smishing", 1.4),
    ("spoof", "Phishing_Smishing", 1.2),
    ("spoofed", "Phishing_Smishing", 1.2),
    ("spoofing", "Phishing_Smishing", 1.2),
    ("fake link", "Phishing_Smishing", 1.3),
    ("fake login", "Phishing_Smishing", 1.3),
    ("fake email", "Phishing_Smishing", 1.3),
    ("fake text", "Phishing_Smishing", 1.3),
    ("login page", "Phishing_Smishing", 1.1),
    ("text message scam", "Phishing_Smishing", 1.4),
    ("sms scam", "Phishing_Smishing", 1.4),
    ("sms message", "Phishing_Smishing", 1.1),
    ("verify your account", "Phishing_Smishing", 1.3),
    ("verify your identity", "Phishing_Smishing", 1.3),
    ("confirm your details", "Phishing_Smishing", 1.3),
    ("urgent action", "Phishing_Smishing", 1.1),
    ("clicked a link", "Phishing_Smishing", 1.2),
    ("malicious link", "Phishing_Smishing", 1.3),
    ("security alert", "Phishing_Smishing", 1.0),
    ("social engineering", "Phishing_Smishing", 1.3),
    ("malware", "Phishing_Smishing", 1.2),
    ("trojan", "Phishing_Smishing", 1.2),
    ("data breach", "Phishing_Smishing", 1.2),
    ("suspicious email", "Phishing_Smishing", 1.3),
    ("suspicious text", "Phishing_Smishing", 1.3),
    ("suspicious link", "Phishing_Smishing", 1.3),
    ("phishing email", "Phishing_Smishing", 1.5),
    ("phishing text", "Phishing_Smishing", 1.5),
    ("remote access", "Phishing_Smishing", 1.2),
    ("credentials stolen", "Phishing_Smishing", 1.3),
    ("credentials compromised", "Phishing_Smishing", 1.3),
    ("account credentials", "Phishing_Smishing", 1.2),
    ("router hacked", "Phishing_Smishing", 1.3),
    ("router hijacked", "Phishing_Smishing", 1.3),
    ("malicious package", "Phishing_Smishing", 1.2),
    ("npm attack", "Phishing_Smishing", 1.2),
    ("supply chain attack", "Phishing_Smishing", 1.3),

    # --- Identity Theft ---
    ("identity theft", "Identity_Theft", 1.5),
    ("stolen identity", "Identity_Theft", 1.5),
    ("synthetic identity", "Identity_Theft", 1.5),
    ("identity fraud", "Identity_Theft", 1.5),
    ("identity document", "Identity_Theft", 1.3),
    ("opened an account", "Identity_Theft", 1.2),
    ("someone opened", "Identity_Theft", 1.2),
    ("fraudulent account", "Identity_Theft", 1.3),
    ("credit freeze", "Identity_Theft", 1.3),
    ("fraud alert", "Identity_Theft", 1.2),
    ("social security number", "Identity_Theft", 1.5),
    ("ssn", "Identity_Theft", 1.4),
    ("drivers license", "Identity_Theft", 1.3),
    ("driver's license", "Identity_Theft", 1.3),
    ("background check", "Identity_Theft", 1.1),
    ("tax fraud", "Identity_Theft", 1.3),
    ("tax identity theft", "Identity_Theft", 1.5),
    ("someone used my", "Identity_Theft", 1.2),
    ("used my information", "Identity_Theft", 1.3),
    ("used my identity", "Identity_Theft", 1.4),
    ("data stolen", "Identity_Theft", 1.3),

    # --- Benefits Fraud ---
    ("government benefits fraud", "Benefits_Fraud", 1.5),
    ("medicaid fraud", "Benefits_Fraud", 1.5),
    ("pandemic relief fraud", "Benefits_Fraud", 1.5),
    ("benefits fraud", "Benefits_Fraud", 1.4),
    ("unemployment fraud", "Benefits_Fraud", 1.4),
    ("irs fraud", "Benefits_Fraud", 1.4),
    ("tax refund fraud", "Benefits_Fraud", 1.4),
    ("stimulus fraud", "Benefits_Fraud", 1.4),
    ("welfare fraud", "Benefits_Fraud", 1.3),

    # --- Payment Scams / P2P ---
    ("zelle", "Payment_Scams_P2P", 1.3),
    ("cash app", "Payment_Scams_P2P", 1.3),
    ("cashapp", "Payment_Scams_P2P", 1.3),
    ("venmo", "Payment_Scams_P2P", 1.3),
    ("paypal", "Payment_Scams_P2P", 1.1),
    ("wire transfer", "Payment_Scams_P2P", 1.3),
    ("international wire", "Payment_Scams_P2P", 1.4),
    ("bank transfer", "Payment_Scams_P2P", 1.2),
    ("peer-to-peer payment", "Payment_Scams_P2P", 1.4),
    ("p2p payment", "Payment_Scams_P2P", 1.4),
    ("refund scam", "Payment_Scams_P2P", 1.3),
    ("chargeback", "Payment_Scams_P2P", 1.2),
    ("unauthorized transaction", "Payment_Scams_P2P", 1.3),
    ("unauthorized charge", "Payment_Scams_P2P", 1.3),
    ("payment reversed", "Payment_Scams_P2P", 1.2),
    ("prepaid card", "Payment_Scams_P2P", 1.2),
    ("gift card scam", "Payment_Scams_P2P", 1.4),
    ("debit card fraud", "Payment_Scams_P2P", 1.4),
    ("credit card fraud", "Payment_Scams_P2P", 1.4),
    ("card stolen", "Payment_Scams_P2P", 1.3),
    ("card skimmer", "Payment_Scams_P2P", 1.4),
    ("unauthorized purchase", "Payment_Scams_P2P", 1.3),
    ("fraudulent charge", "Payment_Scams_P2P", 1.3),
    ("fraudulent transaction", "Payment_Scams_P2P", 1.3),
    ("rapid movement of funds", "Payment_Scams_P2P", 1.4),
    ("round dollar transactions", "Payment_Scams_P2P", 1.3),
    ("atm withdrawal", "Payment_Scams_P2P", 1.2),
    ("cash withdrawal", "Payment_Scams_P2P", 1.2),

    # --- Check Fraud ---
    ("check fraud", "Check_Fraud", 1.5),
    ("mail theft", "Check_Fraud", 1.4),
    ("stolen check", "Check_Fraud", 1.5),
    ("stolen checks", "Check_Fraud", 1.5),
    ("check washing", "Check_Fraud", 1.5),
    ("fraudulent check", "Check_Fraud", 1.5),
    ("altered check", "Check_Fraud", 2.0),
    ("check was altered", "Check_Fraud", 2.0),
    ("check altered", "Check_Fraud", 1.8),
    ("forged check", "Check_Fraud", 2.0),
    ("forged a check", "Check_Fraud", 2.0),
    ("someone forged", "Check_Fraud", 1.5),
    ("forged document", "BEC_Business_Email_Compromise", 1.3),
    ("fabricated records", "BEC_Business_Email_Compromise", 1.3),
    ("fake documentation", "BEC_Business_Email_Compromise", 1.3),

    # --- Crypto Fraud ---
    ("cryptocurrency", "Crypto_Fraud", 1.5),
    ("virtual currency", "Crypto_Fraud", 1.4),
    ("crypto exchange", "Crypto_Fraud", 1.4),
    ("crypto wallet", "Crypto_Fraud", 1.4),
    ("digital wallet", "Crypto_Fraud", 1.3),
    ("wallet address", "Crypto_Fraud", 1.4),
    ("bitcoin address", "Crypto_Fraud", 1.4),
    ("ethereum address", "Crypto_Fraud", 1.4),
    ("bitcoin", "Crypto_Fraud", 1.3),
    ("ethereum", "Crypto_Fraud", 1.3),
    ("metamask", "Crypto_Fraud", 1.3),
    ("coinbase", "Crypto_Fraud", 1.2),
    ("binance", "Crypto_Fraud", 1.2),
    ("seed phrase", "Crypto_Fraud", 1.5),
    ("recovery phrase", "Crypto_Fraud", 1.5),
    ("airdrop scam", "Crypto_Fraud", 1.4),
    ("rug pull", "Crypto_Fraud", 1.5),
    ("rugpull", "Crypto_Fraud", 1.5),
    ("wallet drained", "Crypto_Fraud", 1.5),
    ("pig butchering", "Crypto_Fraud", 1.5),
    ("crypto scam", "Crypto_Fraud", 1.5),
    ("bitcoin scam", "Crypto_Fraud", 1.5),
    ("nft scam", "Crypto_Fraud", 1.4),
    ("defi scam", "Crypto_Fraud", 1.4),
    ("smart contract", "Crypto_Fraud", 1.2),
    ("crypto investment", "Crypto_Fraud", 1.3),

    # --- Tech Support Scam ---
    ("tech support scam", "Tech_Support_Scam", 1.5),
    ("tech support fraud", "Tech_Support_Scam", 1.5),
    ("microsoft support", "Tech_Support_Scam", 1.4),
    ("apple support", "Tech_Support_Scam", 1.4),
    ("norton support", "Tech_Support_Scam", 1.4),
    ("mcafee support", "Tech_Support_Scam", 1.4),
    ("geek squad", "Tech_Support_Scam", 1.4),
    ("best buy scam", "Tech_Support_Scam", 1.3),
    ("pop up", "Tech_Support_Scam", 1.2),
    ("popup", "Tech_Support_Scam", 1.2),
    ("virus alert", "Tech_Support_Scam", 1.3),
    ("virus warning", "Tech_Support_Scam", 1.3),
    ("computer infected", "Tech_Support_Scam", 1.3),
    ("anydesk", "Tech_Support_Scam", 1.4),
    ("teamviewer", "Tech_Support_Scam", 1.4),
    ("support number", "Tech_Support_Scam", 1.2),
    ("call this number", "Tech_Support_Scam", 1.2),
    ("ransomware", "Tech_Support_Scam", 1.3),
    ("subscription renewal", "Tech_Support_Scam", 1.2),
    ("billing alert", "Tech_Support_Scam", 1.2),
    ("subscription scam", "Tech_Support_Scam", 1.4),
    ("fake invoice", "Tech_Support_Scam", 1.3),
    ("fake receipt", "Tech_Support_Scam", 1.3),
    ("fake charge", "Tech_Support_Scam", 1.2),
    ("apple pay scam", "Tech_Support_Scam", 1.4),
    ("fake microsoft", "Tech_Support_Scam", 1.4),
    ("fake apple", "Tech_Support_Scam", 1.4),
    ("fake norton", "Tech_Support_Scam", 1.4),
    ("restoro", "Tech_Support_Scam", 1.4),
    ("reimage", "Tech_Support_Scam", 1.4),

    # --- Investment Scam ---
    ("investment scam", "Investment_Scam", 1.5),
    ("forex scam", "Investment_Scam", 1.5),
    ("guaranteed returns", "Investment_Scam", 1.5),
    ("guaranteed profit", "Investment_Scam", 1.5),
    ("trading platform scam", "Investment_Scam", 1.5),
    ("ponzi", "Investment_Scam", 1.5),
    ("pyramid scheme", "Investment_Scam", 1.5),
    ("romance scam", "Investment_Scam", 1.4),
    ("pig butchering", "Investment_Scam", 1.5),
    ("imposter scam", "Investment_Scam", 1.4),
    ("impersonation", "Investment_Scam", 1.3),
    ("imposter", "Investment_Scam", 1.3),
    ("lottery scam", "Investment_Scam", 1.4),
    ("charity fraud", "Investment_Scam", 1.4),
    ("fake job", "Investment_Scam", 1.3),
    ("job scam", "Investment_Scam", 1.3),
    ("work from home scam", "Investment_Scam", 1.4),
    ("recruitment scam", "Investment_Scam", 1.3),
    ("advance fee", "Investment_Scam", 1.4),
    ("urgent payment request", "Investment_Scam", 1.3),
    ("nigerian prince", "Investment_Scam", 1.4),
    ("419 scam", "Investment_Scam", 1.4),
    ("dating app scam", "Investment_Scam", 1.4),
    ("online dating scam", "Investment_Scam", 1.4),
    ("fake recruiter", "Investment_Scam", 1.3),
    ("fake giveaway", "Investment_Scam", 1.3),
    ("prize scam", "Investment_Scam", 1.3),
    ("sweepstakes scam", "Investment_Scam", 1.3),
    ("pch scam", "Investment_Scam", 1.3),
    ("dating app", "Investment_Scam", 1.2),
    ("met online", "Investment_Scam", 1.2),
    ("online friend", "Investment_Scam", 1.2),
    ("asked for money", "Investment_Scam", 1.2),
    ("send money", "Investment_Scam", 1.1),
    ("pay upfront", "Investment_Scam", 1.2),
    ("task scam", "Investment_Scam", 1.3),
    ("job offer scam", "Investment_Scam", 1.4),
    ("telegram group", "Investment_Scam", 1.2),
    ("whatsapp group", "Investment_Scam", 1.2),
    ("crypto professor", "Investment_Scam", 1.4),
    ("pig butcher", "Investment_Scam", 1.5),

    # --- BEC: Business Email Compromise ---
    ("business email compromise", "BEC_Business_Email_Compromise", 1.5),
    ("bec", "BEC_Business_Email_Compromise", 1.3),
    ("invoice fraud", "BEC_Business_Email_Compromise", 1.5),
    ("fraudulent invoice", "BEC_Business_Email_Compromise", 1.5),
    ("vendor payment", "BEC_Business_Email_Compromise", 1.4),
    ("wire instructions", "BEC_Business_Email_Compromise", 1.5),
    ("payment instructions", "BEC_Business_Email_Compromise", 1.4),
    ("accounts payable", "BEC_Business_Email_Compromise", 1.3),
    ("changed banking", "BEC_Business_Email_Compromise", 1.4),
    ("email compromise", "BEC_Business_Email_Compromise", 1.5),
    ("ceo fraud", "BEC_Business_Email_Compromise", 1.5),
    ("impersonating ceo", "BEC_Business_Email_Compromise", 1.5),
    ("executive impersonation", "BEC_Business_Email_Compromise", 1.4),
    ("vendor impersonation", "BEC_Business_Email_Compromise", 1.4),
    ("false claim", "BEC_Business_Email_Compromise", 1.2),

    # --- Money Laundering ---
    ("money laundering", "Money_Laundering", 1.5),
    ("shell company", "Money_Laundering", 1.5),
    ("shell companies", "Money_Laundering", 1.5),
    ("shell corporation", "Money_Laundering", 1.5),
    ("front company", "Money_Laundering", 1.5),
    ("money mule", "Money_Laundering", 1.5),
    ("money mules", "Money_Laundering", 1.5),
    ("mule account", "Money_Laundering", 1.5),
    ("layering", "Money_Laundering", 1.3),
    ("structuring", "Money_Laundering", 1.4),
    ("smurfing", "Money_Laundering", 1.4),
    ("placement", "Money_Laundering", 1.3),
    ("beneficial owner", "Money_Laundering", 1.3),
    ("ultimate beneficial owner", "Money_Laundering", 1.4),
    ("ubo", "Money_Laundering", 1.3),

    # --- Sanctions ---
    ("sanctions evasion", "Sanctions", 1.5),
    ("sanctioned entity", "Sanctions", 1.5),
    ("export control violation", "Sanctions", 1.5),
    ("ofac", "Sanctions", 1.4),
    ("sanctioned country", "Sanctions", 1.4),

    # --- Terrorist Financing ---
    ("terrorist financing", "Terrorist_Financing", 1.5),
    ("terrorist organization", "Terrorist_Financing", 1.5),
    ("financing terrorism", "Terrorist_Financing", 1.5),

    # --- Human Trafficking ---
    ("human trafficking", "Human_Trafficking", 1.5),
    ("labor trafficking", "Human_Trafficking", 1.5),
    ("sex trafficking", "Human_Trafficking", 1.5),
    ("forced labor", "Human_Trafficking", 1.4),
    ("trafficking victim", "Human_Trafficking", 1.4),

    # --- Elder Fraud ---
    ("grandparent scam", "Elder_Fraud", 1.5),
    ("grandchild scam", "Elder_Fraud", 1.5),
    ("elder fraud", "Elder_Fraud", 1.5),
    ("elder abuse", "Elder_Fraud", 1.4),
    ("senior scam", "Elder_Fraud", 1.4),
    ("targeting elderly", "Elder_Fraud", 1.4),
    ("targeting seniors", "Elder_Fraud", 1.4),
    ("fake va benefits", "Elder_Fraud", 1.5),
    ("va benefit scam", "Elder_Fraud", 1.5),
    ("pension scam", "Elder_Fraud", 1.4),
    ("retirement scam", "Elder_Fraud", 1.4),
    ("caregiver fraud", "Elder_Fraud", 1.4),
    ("power of attorney fraud", "Elder_Fraud", 1.5),
    ("nursing home fraud", "Elder_Fraud", 1.4),
    ("medicare fraud", "Elder_Fraud", 1.4),
    ("social security scam", "Elder_Fraud", 1.4),
    ("ssa scam", "Elder_Fraud", 1.4),
    ("irs impersonation", "Elder_Fraud", 1.4),
    ("fake irs", "Elder_Fraud", 1.4),
    ("fake social security", "Elder_Fraud", 1.4),
    ("grandma scam", "Elder_Fraud", 1.5),
    ("grandpa scam", "Elder_Fraud", 1.5),
    ("parents scammed", "Elder_Fraud", 1.3),
    ("father scammed", "Elder_Fraud", 1.3),
    ("mother scammed", "Elder_Fraud", 1.3),

    # --- Military Scam ---
    ("military scam", "Military_Scam", 1.5),
    ("military romance scam", "Military_Scam", 1.5),
    ("fake soldier", "Military_Scam", 1.5),
    ("fake military", "Military_Scam", 1.5),
    ("pretending to be military", "Military_Scam", 1.5),
    ("claiming to be deployed", "Military_Scam", 1.5),
    ("deployed soldier scam", "Military_Scam", 1.5),
    ("va fraud", "Military_Scam", 1.5),
    ("veterans affairs fraud", "Military_Scam", 1.5),
    ("bah fraud", "Military_Scam", 1.5),
    ("basic allowance housing fraud", "Military_Scam", 1.5),
    ("military charity fraud", "Military_Scam", 1.4),
    ("fake veteran", "Military_Scam", 1.4),
    ("stolen valor", "Military_Scam", 1.3),
    ("military benefit fraud", "Military_Scam", 1.4),
    ("tricare fraud", "Military_Scam", 1.4),
    ("military member scammed", "Military_Scam", 1.4),
    ("service member scam", "Military_Scam", 1.4),
    ("veteran scam", "Military_Scam", 1.4),
    ("military pay scam", "Military_Scam", 1.4),
    ("deployment scam", "Military_Scam", 1.4),

    # --- Data Breach ---
    ("data breach", "Data_Breach", 1.5),
    ("data leak", "Data_Breach", 1.5),
    ("data leaked", "Data_Breach", 1.5),
    ("personal data exposed", "Data_Breach", 1.5),
    ("personal information leaked", "Data_Breach", 1.5),
    ("credentials exposed", "Data_Breach", 1.4),
    ("credentials leaked", "Data_Breach", 1.4),
    ("password exposed", "Data_Breach", 1.4),
    ("breach notification", "Data_Breach", 1.4),
    ("have i been pwned", "Data_Breach", 1.4),
    ("dark web", "Data_Breach", 1.3),
    ("sold on dark web", "Data_Breach", 1.4),
    ("information sold", "Data_Breach", 1.3),
    ("database exposed", "Data_Breach", 1.4),
    ("database leaked", "Data_Breach", 1.4),
    ("millions of records", "Data_Breach", 1.3),
    ("user data exposed", "Data_Breach", 1.4),
    ("security incident", "Data_Breach", 1.2),
    ("router hacked", "Data_Breach", 1.3),
    ("router hijacked", "Data_Breach", 1.3),
    ("network compromised", "Data_Breach", 1.3),
    ("supply chain attack", "Data_Breach", 1.3),
    ("malicious package", "Data_Breach", 1.2),
    ("npm attack", "Data_Breach", 1.2),

    # --- Consumer Billing Fraud ---
    ("unauthorized charge", "Consumer_Billing_Fraud", 1.4),
    ("unauthorized subscription", "Consumer_Billing_Fraud", 1.4),
    ("unexpected charge", "Consumer_Billing_Fraud", 1.3),
    ("mystery charge", "Consumer_Billing_Fraud", 1.3),
    ("fake subscription", "Consumer_Billing_Fraud", 1.4),
    ("subscription fraud", "Consumer_Billing_Fraud", 1.4),
    ("free trial scam", "Consumer_Billing_Fraud", 1.4),
    ("charged without permission", "Consumer_Billing_Fraud", 1.4),
    ("billed without authorization", "Consumer_Billing_Fraud", 1.4),
    ("recurring charge", "Consumer_Billing_Fraud", 1.2),
    ("surprise bill", "Consumer_Billing_Fraud", 1.2),
    ("overcharged", "Consumer_Billing_Fraud", 1.1),
    ("double charged", "Consumer_Billing_Fraud", 1.2),
    ("fake renewal", "Consumer_Billing_Fraud", 1.3),
    ("renewal scam", "Consumer_Billing_Fraud", 1.3),
    ("utility scam", "Consumer_Billing_Fraud", 1.3),
    ("fake utility", "Consumer_Billing_Fraud", 1.3),
    ("electric company scam", "Consumer_Billing_Fraud", 1.3),
    ("cable company scam", "Consumer_Billing_Fraud", 1.3),
    ("internet service scam", "Consumer_Billing_Fraud", 1.3),
    ("landlord scam", "Consumer_Billing_Fraud", 1.3),
    ("rental scam", "Consumer_Billing_Fraud", 1.3),
    ("fake landlord", "Consumer_Billing_Fraud", 1.4),
    ("craigslist scam", "Consumer_Billing_Fraud", 1.3),
    ("facebook marketplace scam", "Consumer_Billing_Fraud", 1.3),
    ("online marketplace scam", "Consumer_Billing_Fraud", 1.3),

    # --- General Scam (catch-all, low weight) ---
    ("scam", "General_Scam", 0.6),
    ("scammer", "General_Scam", 0.6),
    ("fraud", "General_Scam", 0.6),
    ("fraudster", "General_Scam", 0.6),
    ("got scammed", "General_Scam", 0.8),
    ("being scammed", "General_Scam", 0.8),
    ("is this a scam", "General_Scam", 0.8),
    ("suspicious", "General_Scam", 0.5),
    ("stolen", "General_Scam", 0.5),
    ("con artist", "General_Scam", 0.7),
]


# ---------------------------------------------------------------------------
# Tier 2 — Theme word bags
# Individual words that signal a theme.
# Each entry: theme -> ([words], weight_per_hit)
# Only applied when >= 2 word hits to avoid noise.
# ---------------------------------------------------------------------------

THEME_WORD_BAGS = {

    "ATO_Account_Takeover": (
        [
            "hacked", "breach", "compromised", "unauthorized", "login",
            "password", "verification", "authenticate", "lockout", "hijacked",
            "takeover", "reset", "2fa", "mfa", "otp", "authenticator",
            "sessions", "logged", "access", "credentials", "token",
        ],
        0.3
    ),

    "Phishing_Smishing": (
        [
            "phishing", "smishing", "vishing", "spoofed", "impersonating",
            "fake", "link", "clicked", "email", "text", "sms", "message",
            "verify", "credential", "password", "suspicious", "alert",
            "malware", "ransomware", "trojan", "attachment", "urgent",
            "warning", "notification", "bank", "paypal", "amazon",
            "microsoft", "telegram", "whatsapp",
        ],
        0.3
    ),

    "Identity_Theft": (
        [
            "identity", "ssn", "social", "security", "license", "passport",
            "personal", "information", "stolen", "fraud", "impersonation",
            "credit", "freeze", "bureau", "equifax", "experian", "transunion",
            "irs", "tax", "refund", "unemployment", "benefits", "loan",
            "application", "opened", "account", "name",
        ],
        0.3
    ),

    "Benefits_Fraud": (
        [
            "benefits", "medicaid", "unemployment", "irs", "stimulus",
            "pandemic", "relief", "welfare", "government", "fraud",
            "claim", "filing", "tax", "refund", "identity",
        ],
        0.3
    ),

    "Payment_Scams_P2P": (
        [
            "payment", "transfer", "wire", "zelle", "venmo", "cashapp",
            "paypal", "charge", "transaction", "debit", "credit", "card",
            "bank", "account", "money", "sent", "received", "dispute",
            "chargeback", "refund", "unauthorized", "stolen", "check",
            "gift", "prepaid", "apple", "google", "atm", "withdrawal",
        ],
        0.3
    ),

    "Check_Fraud": (
        [
            "check", "cheque", "mail theft", "stolen check", "check washing",
            "forged check", "altered check", "fraudulent check",
            "usps", "mailbox", "routing number", "deposit slip",
            "counterfeit", "fake check", "bad check",
        ],
        0.4  # higher weight per hit since these are more specific
    ),

    "Crypto_Fraud": (
        [
            "crypto", "bitcoin", "ethereum", "blockchain", "wallet",
            "token", "coin", "exchange", "defi", "nft", "mining",
            "invest", "trading", "binance", "coinbase", "metamask",
            "seed", "phrase", "private", "key", "drained", "stolen",
        ],
        0.3
    ),

    "Tech_Support_Scam": (
        [
            "norton", "mcafee", "microsoft", "apple", "geek", "squad",
            "bestbuy", "antivirus", "virus", "malware", "computer",
            "subscription", "renewal", "billing", "charge", "refund",
            "remote", "access", "anydesk", "teamviewer", "popup",
            "alert", "warning", "infected", "support", "helpdesk",
            "restoro", "reimage", "fake", "number", "call",
        ],
        0.3
    ),

    "Investment_Scam": (
        [
            "invest", "investment", "returns", "profit", "trading",
            "forex", "crypto", "stock", "platform", "scheme", "ponzi",
            "pyramid", "recruiter", "job", "hiring", "giveaway", "prize",
            "lottery", "winner", "romance", "dating", "love", "instagram",
            "telegram", "whatsapp", "coach", "mentor", "signal",
            "guaranteed", "passive", "income", "imposter", "impersonation",
        ],
        0.3
    ),

    "BEC_Business_Email_Compromise": (
        [
            "invoice", "payment", "wire", "vendor", "supplier", "ceo",
            "executive", "impersonation", "email", "business", "company",
            "account", "banking", "instructions", "transfer", "request",
            "urgent", "confidential", "payroll", "employee",
        ],
        0.3
    ),

    "Money_Laundering": (
        [
            "laundering", "mule", "shell", "structuring", "layering",
            "smurfing", "cash", "transfer", "deposit", "withdraw",
            "account", "beneficial", "owner", "nominee", "offshore",
            "placement", "ubo",
        ],
        0.3
    ),

    "Elder_Fraud": (
        [
            "elderly", "senior", "grandparent", "grandchild", "grandmother",
            "grandfather", "pension", "retirement", "medicare",
            "ssa", "irs", "impersonation", "caregiver", "nursing",
            "attorney", "va", "benefit", "scammed", "scam", "fraud",
            "stolen", "targeted", "parent", "parents",
        ],
        0.3
    ),

    "Military_Scam": (
        [
            "military", "soldier", "deployed", "deployment", "veteran",
            "tricare", "bah", "allowance",
            "army", "navy", "marines", "airforce", "guard",
            "scam", "fraud", "romance", "fake",
            "impersonation", "stolen", "benefits", "overseas",
        ],
        0.3
    ),

    "Data_Breach": (
        [
            "breach", "leak", "leaked", "exposed", "stolen",
            "credentials", "password", "database", "records",
            "dark", "web", "pwned", "hacked", "compromised",
            "notification", "millions", "router", "hijacked",
            "malicious", "attack", "personal", "information",
        ],
        0.3
    ),

    "Consumer_Billing_Fraud": (
        [
            "charged", "billing", "subscription", "unauthorized", "unexpected",
            "renewal", "overcharged", "double", "charge",
            "utility", "landlord", "rental", "marketplace",
            "craigslist", "etsy", "scam", "fraud",
            "fake", "invoice", "dispute", "cancel", "refund",
        ],
        0.3
    ),
}


# ---------------------------------------------------------------------------
# Fraud signal dictionary
# Used for explainability — which signals triggered a classification.
# ---------------------------------------------------------------------------

FRAUD_SIGNALS = [

    # Entity / organization
    ("shell company", "entity_signal"),
    ("shell companies", "entity_signal"),
    ("front company", "entity_signal"),
    ("beneficial owner", "ownership_signal"),
    ("ultimate beneficial owner", "ownership_signal"),
    ("ubo", "ownership_signal"),

    # Transaction patterns
    ("money mule", "transaction_signal"),
    ("money mules", "transaction_signal"),
    ("mule account", "transaction_signal"),
    ("structuring", "transaction_signal"),
    ("layering", "transaction_signal"),
    ("smurfing", "transaction_signal"),
    ("placement", "transaction_signal"),
    ("round dollar transactions", "transaction_signal"),
    ("rapid movement of funds", "transaction_signal"),

    # Payment methods
    ("wire transfer", "payment_signal"),
    ("international wire", "payment_signal"),
    ("cash withdrawal", "payment_signal"),
    ("atm withdrawal", "payment_signal"),
    ("prepaid card", "payment_signal"),
    ("gift card", "payment_signal"),
    ("zelle", "payment_signal"),
    ("venmo", "payment_signal"),
    ("cash app", "payment_signal"),
    ("paypal", "payment_signal"),
    ("peer-to-peer payment", "payment_signal"),
    ("p2p payment", "payment_signal"),

    # Crypto signals
    ("cryptocurrency", "crypto_signal"),
    ("virtual currency", "crypto_signal"),
    ("crypto exchange", "crypto_signal"),
    ("crypto wallet", "crypto_signal"),
    ("digital wallet", "crypto_signal"),
    ("wallet address", "crypto_signal"),
    ("bitcoin address", "crypto_signal"),
    ("ethereum address", "crypto_signal"),
    ("pig butchering", "crypto_signal"),
    ("rug pull", "crypto_signal"),
    ("seed phrase", "crypto_signal"),
    ("recovery phrase", "crypto_signal"),
    ("wallet drained", "crypto_signal"),

    # Identity / account abuse
    ("identity theft", "identity_signal"),
    ("stolen identity", "identity_signal"),
    ("synthetic identity", "identity_signal"),
    ("account takeover", "identity_signal"),
    ("credential theft", "identity_signal"),
    ("compromised account", "identity_signal"),
    ("unauthorized access", "identity_signal"),
    ("social security number", "identity_signal"),
    ("ssn", "identity_signal"),

    # Cyber / attack methods
    ("phishing", "cyber_signal"),
    ("smishing", "cyber_signal"),
    ("vishing", "cyber_signal"),
    ("malware", "cyber_signal"),
    ("ransomware", "cyber_signal"),
    ("trojan", "cyber_signal"),
    ("data breach", "cyber_signal"),
    ("credential stuffing", "cyber_signal"),
    ("social engineering", "cyber_signal"),
    ("remote access", "cyber_signal"),
    ("business email compromise", "cyber_signal"),

    # Document / claim fraud
    ("false claim", "document_signal"),
    ("false claims", "document_signal"),
    ("fraudulent invoice", "document_signal"),
    ("fake documentation", "document_signal"),
    ("forged document", "document_signal"),
    ("fabricated records", "document_signal"),
    ("fraudulent check", "document_signal"),
    ("altered check", "document_signal"),

    # Scam behavior
    ("imposter", "scam_signal"),
    ("impersonation", "scam_signal"),
    ("romance scam", "scam_signal"),
    ("investment scam", "scam_signal"),
    ("lottery scam", "scam_signal"),
    ("gift card scam", "scam_signal"),
    ("advance fee", "scam_signal"),
    ("urgent payment request", "scam_signal"),

    # Contact / channel signals
    ("text message", "contact_signal"),
    ("sms message", "contact_signal"),
    ("telegram", "contact_signal"),
    ("whatsapp", "contact_signal"),
    ("phone number", "contact_signal"),
    ("email address", "contact_signal"),
]
