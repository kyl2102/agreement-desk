"""The search test questions, shared by 06e (regex grading) and 06g (LLM judge)."""

# (question, relevance rule: ALL patterns must match the chunk text, case-insensitive)
TESTS = [
    ("how much warning to end the deal early",       [r"terminat", r"convenience|without cause|for any reason|at any time", r"notice"]),
    ("who pays the lawyers if we fight",              [r"attorneys?'? ?fees|prevailing party|legal fees"]),
    ("can they hand this contract to someone else",   [r"\bassign", r"consent"]),
    ("what if a hurricane stops them from performing", [r"force majeure|acts? of god"]),
    ("keeping each other's secrets",                  [r"confidential", r"disclos"]),
    ("stop poaching each other's staff",              [r"solicit", r"employ"]),
    ("can we check their books",                      [r"audit|inspect", r"records|books"]),
    ("which state's law applies",                     [r"governed by|governing law|laws of the state"]),
    ("Section 14.A",                                  [r"\b14\.a\b"]),
    ("$150,000",                                      [r"150,000"]),
]

# DRAFT (Claude, Oct 16): 30 more plain-English questions, graded by the LLM judge only (no regex rules needed).
# Ken: edit, delete or add. Phrase them the way a business person would ask, not in contract terms.
JUDGE_ONLY = [
    "does the contract renew by itself",
    "can either side get out if the other goes bankrupt",
    "are we the only ones allowed to sell this",
    "can they compete with us after it ends",
    "who owns what we build together",
    "do they have to protect us if a customer sues",
    "how fast do we have to pay invoices",
    "what happens if we pay late",
    "can prices go up during the contract",
    "how long does the contract last",
    "is there a minimum amount we have to buy",
    "can they use our logo",
    "do we have to buy insurance",
    "where do we go to court if there's a dispute",
    "do disputes go to arbitration instead of court",
    "what happens if the company gets bought",
    "can they audit how we use the software",
    "what warranties do they give on the product",
    "is there a cap on what we can be sued for",
    "do they have to keep things confidential after the contract ends",
    "can they subcontract the work",
    "what notice do we need to give for a breach",
    "can we cancel if they keep missing deadlines",
    "do they give us a license to their technology",
    "who pays shipping",
    "what if they deliver broken goods",
    "are there any territory restrictions",
    "can we hire people who worked on the project",
    "how do we send official notices to each other",
    "is there a most favored customer pricing promise",
]
