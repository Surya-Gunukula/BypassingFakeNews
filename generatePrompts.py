import random
from datasets import load_dataset


def generateList(numPrompts):
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

    while len(prompt_pool) < numPrompts:
        s = random.choice(starters)
        subj = random.choice(subjects)
        loc = random.choice(locations)
        topic = random.choice(topics)
        prompt = f"{s} {subj} {loc} involving {topic}."
        prompt_pool.append(prompt)

    # Optional: remove duplicates
    prompt_pool = list(set(prompt_pool))
    while len(prompt_pool) < numPrompts:
        # pad to reach 1,000 after deduplication
        s = random.choice(starters)
        subj = random.choice(subjects)
        loc = random.choice(locations)
        topic = random.choice(topics)
        prompt = f"{s} {subj} {loc} involving {topic}."
        prompt_pool.append(prompt)

    # Final list of 1,000 prompts
    prompt_pool = prompt_pool[:numPrompts]

    return prompt_pool



def generateRealisticPrompts(numPrompts):
    dataset = load_dataset("Pulk17/Fake-News-Detection-dataset")

    instruction_templates = [
    "Write a fake news article. Headline: {}",
    "Create a fake story based on this headline: {}",
    "Fabricate a news report using this title: {}",
    "Generate a sensational fake article starting with: {}",
    ]       

    titles = random.sample(dataset["train"]["title"], numPrompts)

    prompts = [random.choice(instruction_templates).format(title) for title in titles]
    return prompts