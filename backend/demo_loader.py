"""
Endee Vector DB — Complete Demo Data Loader
Run this AFTER starting the server: python main.py
Then run: python demo_loader.py

Loads 3 real-world datasets:
  1. Tech Articles (semantic search demo)
  2. Product Catalog (filtered search demo)
  3. Movie Database (hybrid search demo)
"""

import requests
import random
import math
import time

BASE = "http://localhost:8080"

# ─────────────────────────────────────────────────────────────────────────────
# Tiny pure-python sentence embedder (no ML libs needed!)
# Uses TF-IDF style bag-of-words projected to fixed dim via hashing trick
# ─────────────────────────────────────────────────────────────────────────────

VOCAB = [
    "the","of","and","to","a","in","is","it","you","that","he","was","for",
    "on","are","with","as","his","they","be","at","one","have","this","from",
    "or","had","by","word","but","not","what","all","were","we","when","your",
    "can","said","there","use","an","each","which","she","do","how","their",
    "if","will","up","other","about","out","many","then","them","these","so",
    "some","her","would","make","like","him","into","time","has","look","two",
    "more","write","go","see","number","no","way","could","people","my","than",
    "first","water","been","call","who","oil","its","now","find","long","down",
    "day","did","get","come","made","may","part","over","new","sound","take",
    "only","little","work","know","place","years","live","me","back","give",
    # domain words
    "vector","database","search","embedding","neural","model","training",
    "machine","learning","deep","transformer","attention","language","query",
    "index","retrieval","semantic","dense","sparse","hybrid","filter","score",
    "product","price","category","rating","review","customer","buy","sell",
    "movie","film","director","actor","genre","drama","comedy","action","sci",
    "fi","thriller","romance","horror","documentary","release","year","cast",
    "python","java","code","software","api","cloud","server","data","ai","ml",
    "health","medical","disease","treatment","patient","doctor","hospital",
    "sport","football","basketball","tennis","cricket","player","team","match",
    "science","physics","chemistry","biology","research","experiment","theory",
    "food","recipe","cooking","restaurant","cuisine","ingredient","dish",
    "travel","city","country","hotel","flight","tourism","culture","history",
    "music","song","artist","album","genre","pop","rock","jazz","classical",
    "technology","innovation","startup","business","finance","market","stock",
    "education","school","university","student","teacher","course","degree",
]

def _hash_embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic hash-based embedding — no dependencies needed."""
    words = text.lower().split()
    vec = [0.0] * dim
    for word in words:
        # two hash positions per word for better coverage
        h1 = hash(word) % dim
        h2 = hash(word + "_b") % dim
        weight = 1.0
        if word in VOCAB:
            weight = 2.0  # boost known vocab
        vec[h1] += weight
        vec[h2] += weight * 0.5
    # L2 normalize
    mag = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / mag, 6) for x in vec]

def _sparse(text: str) -> dict:
    """Simple TF sparse vector from text."""
    words = text.lower().split()
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    total = len(words) or 1
    return {w: round(c / total, 4) for w, c in counts.items() if len(w) > 3}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 — Tech Articles (128-dim)
# ─────────────────────────────────────────────────────────────────────────────

TECH_ARTICLES = [
    {"id":"tech-001","title":"Introduction to Vector Databases","text":"Vector databases store high dimensional embeddings and enable semantic similarity search. They power modern AI applications including RAG pipelines and recommendation systems.","category":"databases","tags":["vector","search","AI"],"year":2024,"rating":9.2},
    {"id":"tech-002","title":"Understanding HNSW Algorithm","text":"Hierarchical Navigable Small World graphs provide approximate nearest neighbor search with logarithmic complexity. HNSW is the backbone of most production vector databases.","category":"algorithms","tags":["HNSW","ANN","graphs"],"year":2023,"rating":8.9},
    {"id":"tech-003","title":"RAG Systems with LLMs","text":"Retrieval Augmented Generation combines large language models with vector search to ground responses in factual documents. This reduces hallucinations and improves accuracy.","category":"AI","tags":["RAG","LLM","GPT"],"year":2024,"rating":9.5},
    {"id":"tech-004","title":"Semantic Search vs Keyword Search","text":"Traditional keyword search relies on exact term matching while semantic search uses embeddings to find conceptually similar content. Hybrid approaches combine both for best results.","category":"search","tags":["semantic","BM25","hybrid"],"year":2023,"rating":8.7},
    {"id":"tech-005","title":"Transformer Architecture Deep Dive","text":"The transformer architecture introduced attention mechanisms that revolutionized natural language processing. Self attention allows models to weigh the importance of different words contextually.","category":"AI","tags":["transformers","attention","NLP"],"year":2023,"rating":9.1},
    {"id":"tech-006","title":"Python for Machine Learning","text":"Python dominates the machine learning ecosystem with libraries like PyTorch, TensorFlow, scikit-learn, and HuggingFace. Its simple syntax accelerates research and production deployment.","category":"programming","tags":["python","ML","pytorch"],"year":2024,"rating":8.5},
    {"id":"tech-007","title":"Building Production APIs with FastAPI","text":"FastAPI is a modern Python web framework for building high performance REST APIs. It uses Python type hints for automatic validation and generates OpenAPI documentation automatically.","category":"programming","tags":["fastapi","REST","python"],"year":2024,"rating":9.0},
    {"id":"tech-008","title":"Embedding Models Comparison","text":"Sentence transformers like all-MiniLM, text-embedding-ada, and E5 generate dense vector representations of text. Choosing the right model depends on domain, language, and latency requirements.","category":"AI","tags":["embeddings","sentence-transformers","ada"],"year":2024,"rating":8.8},
    {"id":"tech-009","title":"Cloud Native Databases","text":"Cloud native databases like Aurora, Spanner, and Cosmos DB are designed for horizontal scalability and high availability. They separate compute and storage for elastic scaling.","category":"databases","tags":["cloud","postgres","scalability"],"year":2023,"rating":8.3},
    {"id":"tech-010","title":"Docker for Data Engineers","text":"Docker containers package applications with all dependencies for consistent deployment across environments. Docker Compose orchestrates multi-service applications including databases and APIs.","category":"devops","tags":["docker","containers","devops"],"year":2023,"rating":8.6},
    {"id":"tech-011","title":"Graph Neural Networks","text":"Graph neural networks extend deep learning to graph-structured data enabling node classification link prediction and graph generation tasks across social networks and molecules.","category":"AI","tags":["GNN","graphs","deep-learning"],"year":2024,"rating":9.3},
    {"id":"tech-012","title":"Microservices Architecture Patterns","text":"Microservices decompose applications into small independent services that communicate via APIs. Patterns like circuit breaker event sourcing and CQRS improve resilience and scalability.","category":"architecture","tags":["microservices","API","patterns"],"year":2023,"rating":8.4},
    {"id":"tech-013","title":"Real-time Data Streaming with Kafka","text":"Apache Kafka is a distributed event streaming platform capable of handling trillions of events per day. It enables real time analytics fraud detection and event driven architectures.","category":"databases","tags":["kafka","streaming","events"],"year":2023,"rating":8.9},
    {"id":"tech-014","title":"Reinforcement Learning from Human Feedback","text":"RLHF aligns large language models with human preferences through a reward model trained on human comparisons. This technique was central to training ChatGPT and Claude.","category":"AI","tags":["RLHF","alignment","LLM"],"year":2024,"rating":9.7},
    {"id":"tech-015","title":"Zero Shot Learning in NLP","text":"Zero shot learning enables models to perform tasks without any task specific training examples by leveraging general language understanding and instruction following capabilities.","category":"AI","tags":["zero-shot","NLP","prompting"],"year":2024,"rating":9.0},
    {"id":"tech-016","title":"Kubernetes Cluster Management","text":"Kubernetes automates deployment scaling and management of containerized applications. It provides service discovery load balancing storage orchestration and self healing capabilities.","category":"devops","tags":["kubernetes","k8s","containers"],"year":2023,"rating":8.7},
    {"id":"tech-017","title":"NoSQL Database Design Patterns","text":"NoSQL databases like MongoDB Redis and Cassandra support flexible schemas and horizontal scaling. Key design patterns include denormalization aggregation and event sourcing.","category":"databases","tags":["nosql","mongodb","redis"],"year":2023,"rating":8.2},
    {"id":"tech-018","title":"Attention is All You Need","text":"The original transformer paper by Vaswani et al introduced multi head self attention replacing recurrent networks. This foundational architecture enabled GPT BERT T5 and all modern LLMs.","category":"AI","tags":["paper","transformer","attention"],"year":2023,"rating":9.8},
    {"id":"tech-019","title":"Sparse vs Dense Retrieval","text":"Dense retrieval uses neural embeddings for semantic matching while sparse retrieval like BM25 uses term frequency. Hybrid systems combining both consistently outperform either alone.","category":"search","tags":["BM25","dense","hybrid"],"year":2024,"rating":9.1},
    {"id":"tech-020","title":"Large Scale Recommendation Systems","text":"Modern recommendation systems at Netflix YouTube and Spotify use two stage retrieval and ranking. The retrieval stage uses ANN search over item embeddings for efficiency at scale.","category":"AI","tags":["recommendation","ANN","ranking"],"year":2024,"rating":9.4},
]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 — Product Catalog (128-dim, for filtered search demo)
# ─────────────────────────────────────────────────────────────────────────────

PRODUCTS = [
    {"id":"prod-001","name":"Wireless Noise Cancelling Headphones","desc":"Premium over ear headphones with active noise cancellation 30 hour battery life and Hi-Res audio support","category":"electronics","brand":"Sony","price":299.99,"rating":4.8,"in_stock":True},
    {"id":"prod-002","name":"Mechanical Gaming Keyboard","desc":"Compact TKL mechanical keyboard with RGB backlight Cherry MX switches and programmable macros for gaming","category":"electronics","brand":"Corsair","price":129.99,"rating":4.6,"in_stock":True},
    {"id":"prod-003","name":"4K Ultra HD Monitor","desc":"27 inch 4K IPS display with 144Hz refresh rate HDR600 support and USB-C connectivity for creative professionals","category":"electronics","brand":"LG","price":549.99,"rating":4.7,"in_stock":False},
    {"id":"prod-004","name":"Standing Desk Converter","desc":"Height adjustable desktop riser converts any desk to standing workstation with dual monitor support and keyboard tray","category":"furniture","brand":"FlexiSpot","price":189.99,"rating":4.5,"in_stock":True},
    {"id":"prod-005","name":"Ergonomic Office Chair","desc":"Fully adjustable lumbar support mesh back office chair with armrests headrest and recline function for all day comfort","category":"furniture","brand":"Herman Miller","price":899.99,"rating":4.9,"in_stock":True},
    {"id":"prod-006","name":"Python Programming Cookbook","desc":"Comprehensive guide to advanced Python patterns async programming data structures and real world application development","category":"books","brand":"OReilly","price":49.99,"rating":4.7,"in_stock":True},
    {"id":"prod-007","name":"Portable SSD 2TB","desc":"Ultra fast portable solid state drive with 2000MB/s read speed USB 3.2 interface and rugged shock resistant design","category":"electronics","brand":"Samsung","price":179.99,"rating":4.8,"in_stock":True},
    {"id":"prod-008","name":"Espresso Machine","desc":"Semi automatic espresso machine with 15 bar pump built in grinder milk frother and programmable shot volume","category":"appliances","brand":"Breville","price":699.99,"rating":4.6,"in_stock":False},
    {"id":"prod-009","name":"Yoga Mat Premium","desc":"Non slip extra thick yoga mat with alignment lines carry strap and eco friendly natural rubber material for all yoga styles","category":"sports","brand":"Manduka","price":88.00,"rating":4.9,"in_stock":True},
    {"id":"prod-010","name":"Smart Watch Series 9","desc":"Advanced health tracking smartwatch with ECG blood oxygen GPS always on display and 18 hour battery life","category":"electronics","brand":"Apple","price":399.99,"rating":4.8,"in_stock":True},
    {"id":"prod-011","name":"Air Purifier HEPA","desc":"True HEPA air purifier removes 99.97 percent of particles dust pollen mold and smoke with real time air quality display","category":"appliances","brand":"Dyson","price":349.99,"rating":4.7,"in_stock":True},
    {"id":"prod-012","name":"Whey Protein Powder","desc":"Premium whey protein isolate with 25g protein per serving low sugar naturally flavored available in chocolate vanilla","category":"sports","brand":"Optimum","price":59.99,"rating":4.6,"in_stock":True},
    {"id":"prod-013","name":"Wireless Gaming Mouse","desc":"Ultra lightweight wireless gaming mouse with 25600 DPI sensor 70 hour battery and customizable RGB for FPS gaming","category":"electronics","brand":"Logitech","price":159.99,"rating":4.7,"in_stock":True},
    {"id":"prod-014","name":"Deep Learning Book","desc":"Comprehensive textbook covering neural networks convolutional networks recurrent networks and modern deep learning techniques by Goodfellow","category":"books","brand":"MIT Press","price":79.99,"rating":4.9,"in_stock":True},
    {"id":"prod-015","name":"Coffee Grinder Burr","desc":"Conical burr coffee grinder with 40 grind settings stainless steel burrs for espresso drip and french press brewing","category":"appliances","brand":"Baratza","price":249.99,"rating":4.8,"in_stock":True},
]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3 — Movies (128-dim, for hybrid search demo)
# ─────────────────────────────────────────────────────────────────────────────

MOVIES = [
    {"id":"movie-001","title":"Inception","desc":"A thief who enters people's dreams to steal secrets is given a chance to have his past crimes erased if he can plant an idea in someone's mind","genre":"sci-fi","director":"Christopher Nolan","year":2010,"rating":8.8,"language":"english"},
    {"id":"movie-002","title":"The Shawshank Redemption","desc":"Two imprisoned men bond over a number of years finding solace and eventual redemption through acts of common decency","genre":"drama","director":"Frank Darabont","year":1994,"rating":9.3,"language":"english"},
    {"id":"movie-003","title":"Interstellar","desc":"A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival on a new planet","genre":"sci-fi","director":"Christopher Nolan","year":2014,"rating":8.6,"language":"english"},
    {"id":"movie-004","title":"Parasite","desc":"Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan","genre":"thriller","director":"Bong Joon-ho","year":2019,"rating":8.5,"language":"korean"},
    {"id":"movie-005","title":"The Dark Knight","desc":"Batman raises the stakes in his war on crime when the Joker a criminal mastermind wreaks havoc and chaos on Gotham City","genre":"action","director":"Christopher Nolan","year":2008,"rating":9.0,"language":"english"},
    {"id":"movie-006","title":"Spirited Away","desc":"During her family's move to the suburbs a sullen girl wanders into a world ruled by gods witches and spirits where humans are changed into beasts","genre":"animation","director":"Hayao Miyazaki","year":2001,"rating":8.6,"language":"japanese"},
    {"id":"movie-007","title":"The Godfather","desc":"The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son","genre":"drama","director":"Francis Ford Coppola","year":1972,"rating":9.2,"language":"english"},
    {"id":"movie-008","title":"Pulp Fiction","desc":"The lives of two mob hitmen a boxer a gangster and his wife and a pair of diner bandits intertwine in four tales of violence and redemption","genre":"thriller","director":"Quentin Tarantino","year":1994,"rating":8.9,"language":"english"},
    {"id":"movie-009","title":"The Matrix","desc":"A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers","genre":"sci-fi","director":"Wachowski Sisters","year":1999,"rating":8.7,"language":"english"},
    {"id":"movie-010","title":"Schindlers List","desc":"In German-occupied Poland during World War II Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution","genre":"drama","director":"Steven Spielberg","year":1993,"rating":9.0,"language":"english"},
    {"id":"movie-011","title":"Your Name","desc":"Two strangers find themselves linked in a bizarre way when a comet falls toward Earth and they begin switching bodies during their sleep","genre":"animation","director":"Makoto Shinkai","year":2016,"rating":8.4,"language":"japanese"},
    {"id":"movie-012","title":"Dune","desc":"A noble family becomes embroiled in a war for control over the galaxys most valuable asset while its scion embarks on a journey to a desert planet","genre":"sci-fi","director":"Denis Villeneuve","year":2021,"rating":8.0,"language":"english"},
    {"id":"movie-013","title":"Everything Everywhere All At Once","desc":"An aging Chinese immigrant is swept up in an insane adventure in which she alone can save the world by exploring other universes","genre":"sci-fi","director":"Daniels","year":2022,"rating":7.8,"language":"english"},
    {"id":"movie-014","title":"RRR","desc":"A fictional story about two legendary revolutionaries and their journey away from home before they started fighting for their country in the 1920s","genre":"action","director":"SS Rajamouli","year":2022,"rating":7.9,"language":"telugu"},
    {"id":"movie-015","title":"3 Idiots","desc":"Two friends are searching for their long lost companion and reminisce about the times they had with him during their college days","genre":"comedy","director":"Rajkumar Hirani","year":2009,"rating":8.4,"language":"hindi"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Loader helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_index(name, dim=128, metric="cosine"):
    try:
        r = requests.post(f"{BASE}/indexes", json={
            "name": name, "dim": dim, "metric": metric,
            "M": 16, "ef_construction": 200
        })
        if r.status_code == 201:
            print(f"  ✅ Created index '{name}' (dim={dim})")
        elif r.status_code == 409:
            print(f"  ℹ️  Index '{name}' already exists — skipping create")
        else:
            print(f"  ⚠️  {r.json()}")
    except Exception as e:
        print(f"  ❌ Cannot reach server: {e}")
        raise SystemExit(1)


def upsert_batch(index_name, documents):
    r = requests.post(f"{BASE}/indexes/{index_name}/upsert", json={"documents": documents})
    if r.status_code == 200:
        try:
            print(f"  ✅ Upserted {r.json()['upserted']} docs into '{index_name}'")
        except Exception:
            print(f"  ✅ Upserted into '{index_name}': {r.text[:100]}")
    else:
        print(f"  ❌ Error {r.status_code}: {r.text[:200]}")


def search_demo(index_name, query_text, filters=None, sparse=None, alpha=0.7, top_k=3):
    vec = _hash_embed(query_text)
    body = {
        "vector": vec, "top_k": top_k, "ef_search": 100,
        "alpha": alpha, "include_payload": True,
        "filters": filters or [],
        "sparse_vector": sparse,
    }
    r = requests.post(f"{BASE}/indexes/{index_name}/search", json=body)
    if r.status_code != 200:
        print(f"  ❌ Search error: {r.json()}")
        return
    data = r.json()
    print(f"\n  🔍 Query: '{query_text}' | mode={data['mode']} | {data['total']} hits in {data['query_time_ms']}ms")
    for i, hit in enumerate(data["hits"]):
        pl = hit.get("payload", {}) or {}
        name = pl.get("title") or pl.get("name") or pl.get("text","")[:50]
        cat  = pl.get("category") or pl.get("genre","")
        print(f"     #{i+1}  [{hit['id']}]  score={hit['score']:.4f}  {name[:45]}  [{cat}]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4 — General Conversation / FAQ
# ─────────────────────────────────────────────────────────────────────────────

CONVERSATIONS = [
    {"id":"conv-001","q":"What is the weather like today?","a":"I can check real-time weather for any city. Weather depends on your location and season.","topic":"weather","intent":"information","difficulty":"easy"},
    {"id":"conv-002","q":"How do I reset my password?","a":"Go to the login page click Forgot Password enter your email and follow the reset link sent to your inbox.","topic":"account","intent":"support","difficulty":"easy"},
    {"id":"conv-003","q":"What are the best restaurants nearby?","a":"I recommend checking Google Maps or Yelp for top-rated restaurants near your location with reviews and ratings.","topic":"food","intent":"recommendation","difficulty":"easy"},
    {"id":"conv-004","q":"How can I improve my productivity?","a":"Use time-blocking prioritize tasks with the Eisenhower matrix take regular breaks and minimize distractions.","topic":"productivity","intent":"advice","difficulty":"medium"},
    {"id":"conv-005","q":"What is the meaning of life?","a":"Philosophers debate this endlessly. Common answers include happiness purpose relationships and contributing to something greater than yourself.","topic":"philosophy","intent":"discussion","difficulty":"hard"},
    {"id":"conv-006","q":"How do I learn a new language fast?","a":"Immersion works best with daily practice language exchange apps watching shows and speaking with native speakers.","topic":"education","intent":"advice","difficulty":"medium"},
    {"id":"conv-007","q":"Can you recommend a good book?","a":"For fiction try Dune or 1984. For self-improvement try Atomic Habits. For tech try Clean Code or Designing Data-Intensive Applications.","topic":"books","intent":"recommendation","difficulty":"easy"},
    {"id":"conv-008","q":"How do I stay healthy and fit?","a":"Exercise 3-5 times per week eat balanced meals with vegetables and protein sleep 7-9 hours stay hydrated and manage stress.","topic":"health","intent":"advice","difficulty":"medium"},
    {"id":"conv-009","q":"What is cryptocurrency and should I invest?","a":"Cryptocurrency is digital currency on a blockchain. Investment is high risk and volatile. Only invest what you can afford to lose.","topic":"finance","intent":"advice","difficulty":"hard"},
    {"id":"conv-010","q":"How do I deal with stress at work?","a":"Set clear boundaries take breaks communicate with your manager practice mindfulness exercise regularly and avoid bringing work home.","topic":"wellness","intent":"advice","difficulty":"medium"},
    {"id":"conv-011","q":"What are some good travel destinations?","a":"Japan for culture Portugal for budget travel New Zealand for nature Thailand for food and beaches Iceland for landscapes.","topic":"travel","intent":"recommendation","difficulty":"easy"},
    {"id":"conv-012","q":"How do I start a small business?","a":"Validate your idea write a business plan register your business set up finances build an online presence and start marketing.","topic":"business","intent":"advice","difficulty":"hard"},
    {"id":"conv-013","q":"What is meditation and how do I start?","a":"Meditation is focused attention training. Start with 5 minutes daily using apps like Headspace focus on your breath and return when distracted.","topic":"wellness","intent":"how-to","difficulty":"easy"},
    {"id":"conv-014","q":"How do I save money on a tight budget?","a":"Track every expense cut subscriptions cook at home use cashback apps buy generic brands and automate small savings transfers.","topic":"finance","intent":"advice","difficulty":"medium"},
    {"id":"conv-015","q":"What programming language should I learn first?","a":"Python is the best first language with simple syntax versatile for web AI data science and automation with huge community support.","topic":"technology","intent":"advice","difficulty":"medium"},
    {"id":"conv-016","q":"How does the internet work?","a":"Your device sends data packets through routers across a global network of cables and servers using TCP/IP protocols to reach websites.","topic":"technology","intent":"explanation","difficulty":"medium"},
    {"id":"conv-017","q":"What are some tips for better sleep?","a":"Keep a consistent schedule avoid screens before bed keep your room cool and dark avoid caffeine after 2pm and try deep breathing.","topic":"health","intent":"advice","difficulty":"easy"},
    {"id":"conv-018","q":"How do I apologize sincerely?","a":"Acknowledge what you did wrong express genuine remorse avoid excuses explain how you will change and give the person space to respond.","topic":"relationships","intent":"advice","difficulty":"medium"},
    {"id":"conv-019","q":"What is climate change and why does it matter?","a":"Climate change is the long-term shift in global temperatures caused by greenhouse gas emissions causing extreme weather and rising seas.","topic":"environment","intent":"explanation","difficulty":"medium"},
    {"id":"conv-020","q":"How do I negotiate a salary raise?","a":"Research market rates document your achievements choose the right moment make a specific ask with data and be prepared to discuss alternatives.","topic":"career","intent":"advice","difficulty":"hard"},
]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 5 — Technical / Software Engineering
# ─────────────────────────────────────────────────────────────────────────────

TECHNICAL = [
    {"id":"tech-t01","title":"REST vs GraphQL APIs","text":"REST uses fixed endpoints returning full resources while GraphQL uses a single endpoint where clients specify exactly what data they need reducing over-fetching and under-fetching.","domain":"api-design","level":"intermediate","tags":"REST,GraphQL,API"},
    {"id":"tech-t02","title":"Database Indexing Strategies","text":"B-tree indexes speed up range queries and equality checks. Hash indexes are fast for equality only. Composite indexes cover multiple columns. Covering indexes include all query columns avoiding table lookups.","domain":"databases","level":"advanced","tags":"SQL,indexing,performance"},
    {"id":"tech-t03","title":"CAP Theorem Explained","text":"The CAP theorem states distributed systems can only guarantee two of three properties Consistency Availability and Partition Tolerance. Most systems choose AP or CP depending on use case.","domain":"distributed-systems","level":"advanced","tags":"CAP,distributed,consistency"},
    {"id":"tech-t04","title":"Docker vs Virtual Machines","text":"Docker containers share the host OS kernel making them lightweight and fast to start. VMs include a full OS guest making them heavier but more isolated. Containers suit microservices VMs suit full isolation.","domain":"devops","level":"intermediate","tags":"Docker,VMs,containers"},
    {"id":"tech-t05","title":"Git Branching Strategies","text":"Gitflow uses feature develop release and hotfix branches. Trunk-based development uses short-lived branches merged frequently. GitHub Flow is simpler with feature branches merged directly to main.","domain":"version-control","level":"intermediate","tags":"Git,branching,workflow"},
    {"id":"tech-t06","title":"SQL vs NoSQL Databases","text":"SQL databases enforce schema and ACID transactions best for structured relational data. NoSQL databases like MongoDB and Cassandra offer flexible schemas and horizontal scaling for unstructured data at scale.","domain":"databases","level":"beginner","tags":"SQL,NoSQL,MongoDB"},
    {"id":"tech-t07","title":"CI/CD Pipeline Design","text":"Continuous Integration merges code frequently with automated testing. Continuous Delivery automates release preparation. Continuous Deployment automatically pushes to production. Tools include Jenkins GitHub Actions and GitLab CI.","domain":"devops","level":"intermediate","tags":"CI/CD,automation,DevOps"},
    {"id":"tech-t08","title":"Microservices vs Monolithic Architecture","text":"Monoliths are simpler to develop and deploy initially but harder to scale. Microservices enable independent scaling and deployment but add network complexity and operational overhead.","domain":"architecture","level":"advanced","tags":"microservices,monolith,architecture"},
    {"id":"tech-t09","title":"Big O Notation and Algorithm Complexity","text":"Big O describes worst-case time complexity. O(1) constant O(log n) logarithmic O(n) linear O(n log n) typical for sorting O(n2) quadratic. Space complexity follows the same notation.","domain":"algorithms","level":"intermediate","tags":"algorithms,complexity,BigO"},
    {"id":"tech-t10","title":"OAuth 2.0 Authentication Flow","text":"OAuth 2.0 lets users grant third-party apps access without sharing passwords. The authorization code flow redirects users to an auth server which returns a code exchanged for tokens by the backend.","domain":"security","level":"advanced","tags":"OAuth,security,authentication"},
    {"id":"tech-t11","title":"Memory Management in Python","text":"Python uses reference counting with a cyclic garbage collector. Objects are freed when reference count reaches zero. The GIL prevents true multi-threading for CPU tasks making multiprocessing preferred for CPU-bound work.","domain":"python","level":"advanced","tags":"Python,memory,GIL"},
    {"id":"tech-t12","title":"Load Balancing Algorithms","text":"Round robin distributes requests evenly. Least connections sends to the least busy server. IP hash routes same client to same server for session persistence. Weighted round robin accounts for server capacity differences.","domain":"infrastructure","level":"intermediate","tags":"load-balancing,networking,scalability"},
    {"id":"tech-t13","title":"WebSockets vs HTTP Polling","text":"HTTP polling repeatedly requests updates causing latency and server load. WebSockets establish a persistent bidirectional connection enabling real-time push from server to client with much lower overhead.","domain":"networking","level":"intermediate","tags":"WebSocket,HTTP,real-time"},
    {"id":"tech-t14","title":"SOLID Principles in OOP","text":"Single Responsibility Open-Closed Liskov Substitution Interface Segregation and Dependency Inversion. These five principles guide object-oriented design for maintainable extensible and testable code.","domain":"software-design","level":"intermediate","tags":"SOLID,OOP,design-patterns"},
    {"id":"tech-t15","title":"Caching Strategies","text":"Cache-aside loads data on miss. Write-through updates cache and DB together. Write-behind updates cache first and DB asynchronously. Read-through uses cache as primary with automatic DB fallback. TTL controls expiry.","domain":"performance","level":"advanced","tags":"caching,Redis,performance"},
    {"id":"tech-t16","title":"TCP vs UDP Protocols","text":"TCP provides reliable ordered delivery with handshaking and error correction suitable for web and file transfer. UDP is faster with no guarantees suitable for video streaming gaming and DNS lookups.","domain":"networking","level":"intermediate","tags":"TCP,UDP,networking"},
    {"id":"tech-t17","title":"Test Driven Development","text":"TDD follows a red-green-refactor cycle. Write a failing test first write minimal code to pass it then refactor. This ensures high test coverage better design and confidence when changing code.","domain":"testing","level":"intermediate","tags":"TDD,testing,quality"},
    {"id":"tech-t18","title":"Event-Driven Architecture","text":"Components communicate through events rather than direct calls. Producers emit events to a message broker like Kafka or RabbitMQ. Consumers subscribe and react asynchronously enabling loose coupling and scalability.","domain":"architecture","level":"advanced","tags":"events,Kafka,messaging"},
    {"id":"tech-t19","title":"Kubernetes Pod Lifecycle","text":"Pods go through Pending Running Succeeded or Failed states. Init containers run first. Liveness probes restart unhealthy containers. Readiness probes control traffic routing. Resource limits prevent noisy neighbors.","domain":"kubernetes","level":"advanced","tags":"Kubernetes,k8s,containers"},
    {"id":"tech-t20","title":"Hashing and Encryption Differences","text":"Hashing is one-way transformation used for passwords and integrity checks using SHA-256 or bcrypt. Encryption is reversible using a key used for data confidentiality. Never use MD5 or SHA-1 for security.","domain":"security","level":"intermediate","tags":"hashing,encryption,security"},
]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 6 — Natural Language Processing
# ─────────────────────────────────────────────────────────────────────────────

NLP_DATASET = [
    {"id":"nlp-001","title":"Tokenization in NLP","text":"Tokenization splits text into words subwords or characters. Word tokenization splits on whitespace. BPE and WordPiece create subword tokens handling unknown words. Character-level models process individual characters.","task":"preprocessing","model_type":"foundational","year":2019},
    {"id":"nlp-002","title":"Word Embeddings Word2Vec","text":"Word2Vec learns dense vector representations of words from co-occurrence patterns. CBOW predicts center word from context. Skip-gram predicts context from center word. Similar words cluster in vector space.","task":"embeddings","model_type":"static","year":2013},
    {"id":"nlp-003","title":"BERT Bidirectional Transformers","text":"BERT pre-trains on masked language modeling and next sentence prediction using bidirectional attention. Fine-tuning on downstream tasks achieves state-of-the-art on question answering classification and NER.","task":"language-modeling","model_type":"transformer","year":2018},
    {"id":"nlp-004","title":"Named Entity Recognition","text":"NER identifies and classifies named entities like persons organizations locations and dates in text. Models use IOB tagging scheme. BERT-based NER achieves high F1 scores on standard benchmarks.","task":"NER","model_type":"sequence-labeling","year":2020},
    {"id":"nlp-005","title":"Sentiment Analysis Techniques","text":"Rule-based sentiment uses lexicons like VADER. ML approaches use logistic regression on TF-IDF features. Deep learning uses LSTM or transformer models. Aspect-based sentiment extracts opinions per attribute.","task":"classification","model_type":"various","year":2021},
    {"id":"nlp-006","title":"Machine Translation with Seq2Seq","text":"Sequence-to-sequence models encode input to a context vector and decode to target language. Attention mechanisms align source and target tokens. Transformer-based models like MarianMT achieve near-human translation quality.","task":"translation","model_type":"seq2seq","year":2017},
    {"id":"nlp-007","title":"Text Summarization Methods","text":"Extractive summarization selects key sentences from the source. Abstractive summarization generates new sentences using encoder-decoder models like BART and T5. Evaluation uses ROUGE scores for overlap measurement.","task":"summarization","model_type":"transformer","year":2020},
    {"id":"nlp-008","title":"Question Answering Systems","text":"Extractive QA finds answer spans in passages using models like BERT fine-tuned on SQuAD. Generative QA produces free-form answers. Open-domain QA combines retrieval with reading comprehension for knowledge-intensive tasks.","task":"question-answering","model_type":"transformer","year":2019},
    {"id":"nlp-009","title":"Coreference Resolution","text":"Coreference resolution links mentions referring to the same entity across a document. Neural models use span representations and antecedent scoring. Critical for document understanding and information extraction pipelines.","task":"coreference","model_type":"span-based","year":2018},
    {"id":"nlp-010","title":"Text Classification with Fine-tuning","text":"Pre-trained language models fine-tuned on labeled data achieve strong classification results with minimal data. Adding a classification head on top of BERT CLS token and training end-to-end works well for most tasks.","task":"classification","model_type":"transformer","year":2019},
    {"id":"nlp-011","title":"GPT Autoregressive Language Models","text":"GPT models predict the next token autoregressively using causal attention. Scaling laws show performance improves predictably with more parameters and data. GPT-3 demonstrated few-shot learning without fine-tuning.","task":"language-modeling","model_type":"autoregressive","year":2020},
    {"id":"nlp-012","title":"Relation Extraction from Text","text":"Relation extraction identifies semantic relationships between entities. Approaches include pattern-based supervised classification and distant supervision using knowledge bases. BERT achieves strong results on TACRED and SemEval benchmarks.","task":"information-extraction","model_type":"transformer","year":2020},
    {"id":"nlp-013","title":"Dependency Parsing","text":"Dependency parsing identifies grammatical relationships between words forming a tree structure. Transition-based parsers use shift-reduce operations. Graph-based parsers score all arcs globally. Used in information extraction and semantic analysis.","task":"parsing","model_type":"graph-based","year":2016},
    {"id":"nlp-014","title":"Zero-Shot and Few-Shot Learning","text":"Zero-shot learning uses natural language task descriptions to generalize without examples. Few-shot learning provides a handful of examples in the prompt. GPT-3 demonstrated remarkable few-shot abilities through in-context learning.","task":"few-shot","model_type":"large-language-model","year":2020},
    {"id":"nlp-015","title":"Retrieval Augmented Generation","text":"RAG combines dense retrieval with generative models to ground responses in documents. A retriever fetches relevant passages using vector similarity. A generator conditions on retrieved context to produce factual answers reducing hallucination.","task":"question-answering","model_type":"rag","year":2021},
    {"id":"nlp-016","title":"Dialogue Systems and Chatbots","text":"Task-oriented dialogue systems use NLU dialogue management and NLG components. Open-domain chatbots use retrieval or generative approaches. DialoGPT and BlenderBot fine-tune language models on conversation datasets.","task":"dialogue","model_type":"generative","year":2020},
    {"id":"nlp-017","title":"Semantic Textual Similarity","text":"STS measures how similar two sentences are in meaning on a scale of 0 to 5. Sentence-BERT produces sentence embeddings that encode meaning well. Cosine similarity between embeddings correlates strongly with human judgments.","task":"similarity","model_type":"sentence-transformer","year":2019},
    {"id":"nlp-018","title":"Text Generation and Decoding Strategies","text":"Greedy decoding picks the highest probability token. Beam search explores multiple hypotheses. Top-k sampling restricts to k most likely tokens. Nucleus sampling uses a probability threshold. Temperature controls randomness.","task":"generation","model_type":"autoregressive","year":2020},
    {"id":"nlp-019","title":"Cross-lingual Transfer Learning","text":"Multilingual models like mBERT and XLM-RoBERTa train on 100 plus languages enabling zero-shot cross-lingual transfer. Fine-tuning on English data transfers to other languages through shared representations in multilingual embedding space.","task":"cross-lingual","model_type":"multilingual","year":2020},
    {"id":"nlp-020","title":"Instruction Tuning and RLHF","text":"Instruction tuning fine-tunes LLMs on diverse task instructions improving generalization. RLHF trains a reward model on human preferences then optimizes the policy using PPO. Used to align ChatGPT and Claude to follow instructions safely.","task":"alignment","model_type":"large-language-model","year":2022},
]

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ENDEE VECTOR DB — Demo Data Loader")
    print("="*60)

    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        print(f"\n✅ Server is up: {r.json()}")
    except Exception:
        print("\n❌ Server not running! Start it first:\n   python main.py\n")
        raise SystemExit(1)

    # ── Dataset 1: Tech Articles ─────────────────────────────────────────────
    print("\n📚 Loading Dataset 1: Tech Articles (20 docs)")
    create_index("tech-articles", dim=128)
    docs = []
    for art in TECH_ARTICLES:
        text = art["title"] + " " + art["text"]
        docs.append({"id": art["id"], "vector": _hash_embed(text), "sparse_vector": _sparse(text),
            "payload": {"title": art["title"], "text": art["text"][:120], "category": art["category"],
                        "year": art["year"], "rating": art["rating"], "tags": ", ".join(art["tags"])}})
    upsert_batch("tech-articles", docs)

    # ── Dataset 2: Products ──────────────────────────────────────────────────
    print("\n🛒 Loading Dataset 2: Product Catalog (15 docs)")
    create_index("products", dim=128)
    docs = []
    for p in PRODUCTS:
        text = p["name"] + " " + p["desc"]
        docs.append({"id": p["id"], "vector": _hash_embed(text), "sparse_vector": _sparse(text),
            "payload": {"name": p["name"], "category": p["category"], "brand": p["brand"],
                        "price": p["price"], "rating": p["rating"], "in_stock": p["in_stock"],
                        "desc": p["desc"][:100]}})
    upsert_batch("products", docs)

    # ── Dataset 3: Movies ────────────────────────────────────────────────────
    print("\n🎬 Loading Dataset 3: Movie Database (15 docs)")
    create_index("movies", dim=128)
    docs = []
    for m in MOVIES:
        text = m["title"] + " " + m["desc"]
        docs.append({"id": m["id"], "vector": _hash_embed(text), "sparse_vector": _sparse(text),
            "payload": {"title": m["title"], "genre": m["genre"], "director": m["director"],
                        "year": m["year"], "rating": m["rating"], "language": m["language"],
                        "desc": m["desc"][:100]}})
    upsert_batch("movies", docs)

    # ── Dataset 4: Conversations ─────────────────────────────────────────────
    print("\n💬 Loading Dataset 4: General Conversations (20 docs)")
    create_index("conversations", dim=128)
    docs = []
    for c in CONVERSATIONS:
        text = c["q"] + " " + c["a"]
        docs.append({"id": c["id"], "vector": _hash_embed(text), "sparse_vector": _sparse(text),
            "payload": {"question": c["q"], "answer": c["a"][:120],
                        "topic": c["topic"], "intent": c["intent"], "difficulty": c["difficulty"]}})
    upsert_batch("conversations", docs)

    # ── Dataset 5: Technical ─────────────────────────────────────────────────
    print("\n⚙️  Loading Dataset 5: Technical / Software Engineering (20 docs)")
    create_index("technical", dim=128)
    docs = []
    for t in TECHNICAL:
        text = t["title"] + " " + t["text"]
        docs.append({"id": t["id"], "vector": _hash_embed(text), "sparse_vector": _sparse(text),
            "payload": {"title": t["title"], "text": t["text"][:120],
                        "domain": t["domain"], "level": t["level"], "tags": t["tags"]}})
    upsert_batch("technical", docs)

    # ── Dataset 6: NLP Papers ────────────────────────────────────────────────
    print("\n🧠 Loading Dataset 6: NLP Research Papers (20 docs)")
    create_index("nlp-papers", dim=128)
    docs = []
    for n in NLP_DATASET:
        text = n["title"] + " " + n["text"]
        docs.append({"id": n["id"], "vector": _hash_embed(text), "sparse_vector": _sparse(text),
            "payload": {"title": n["title"], "text": n["text"][:120],
                        "task": n["task"], "model_type": n["model_type"], "year": n["year"]}})
    upsert_batch("nlp-papers", docs)

    print("\n" + "="*60)
    print("  ✅ ALL 6 DATASETS LOADED!")
    print("="*60)
    print("""
  Indexes available:
    • tech-articles   — 20 docs  (AI/tech articles)
    • products        — 15 docs  (product catalog)
    • movies          — 15 docs  (movie database)
    • conversations   — 20 docs  (general Q&A)
    • technical       — 20 docs  (software engineering)
    • nlp-papers      — 20 docs  (NLP research)

  Try in the AI Assistant panel:
    Index: conversations  →  "how to deal with stress at work"
    Index: technical      →  "explain microservices vs monolith"
    Index: nlp-papers     →  "transformer models for text classification"
    Index: products       →  "wireless electronics under $200"
    """)