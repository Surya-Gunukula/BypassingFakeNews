import random

def generateList():
    topics = [
        "health", "technology", "politics", "finance", "education", "science",
        "sports", "climate", "security", "entertainment", "space", "agriculture"
    ]

    starters = [
        "Breaking News:", "Sources report that", "According to officials,", 
        "A new study reveals", "Leaked documents suggest", "Government confirms", 
        "Experts claim", "Reports show", "Scientists discover", "Authorities investigate",
        "Eyewitnesses say", "Data suggests", "Insiders reveal"
    ]

    subjects = [
        "a major outbreak in", "a secret agreement with", "unexpected side effects of", 
        "AI is replacing jobs in", "a new breakthrough in", "fraud in", 
        "illegal surveillance in", "record unemployment in", "a cover-up involving", 
        "billions invested in", "new energy sources found in", "spy networks in"
    ]

    locations = [
        "California", "China", "Texas", "the U.S.", "India", "Germany", 
        "New York", "Antarctica", "Africa", "Russia", "the Middle East", 
        "Silicon Valley", "the Arctic", "Brazil"
    ]

    prompt_pool = []

    while len(prompt_pool) < 1000:
        s = random.choice(starters)
        subj = random.choice(subjects)
        loc = random.choice(locations)
        topic = random.choice(topics)
        prompt = f"{s} {subj} {loc} involving {topic}."
        prompt_pool.append(prompt)

    # Optional: remove duplicates
    prompt_pool = list(set(prompt_pool))
    while len(prompt_pool) < 1000:
        # pad to reach 1,000 after deduplication
        s = random.choice(starters)
        subj = random.choice(subjects)
        loc = random.choice(locations)
        topic = random.choice(topics)
        prompt = f"{s} {subj} {loc} involving {topic}."
        prompt_pool.append(prompt)

    # Final list of 1,000 prompts
    prompt_pool = prompt_pool[:1000]

    return prompt_pool