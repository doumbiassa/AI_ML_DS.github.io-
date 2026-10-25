"""
Generate the Data Wrangling & Engineering bundles (multi-file, multi-format datasets).

Output: wrangling/<bundle-id>/...   one folder per bundle, plus wrangling/<bundle-id>.zip
        wrangling_catalog.json      metadata used by the website
All people, places and organizations are fictional.
Run:  python wrangling_gen.py
"""
import gzip, json, os, shutil, sqlite3, zipfile, textwrap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "wrangling")
CATALOG = []
FIRST = ["Amara", "Jalen", "Sofia", "Kofi", "Maya", "Diego", "Aisha", "Noah", "Zara", "Malik", "Lena", "Omar", "Imani",
         "Lucas", "Nia", "Ravi", "Chloe", "Tariq", "Elena", "Kai", "Ayo", "Mei", "Andre", "Priya", "Jonah", "Fatou", "Leo",
         "Tess", "Idris", "Rosa", "Kwame", "Hana", "Marcus", "Ines", "Tobi", "Grace", "Samir", "Lila", "Ethan", "Adaeze"]
LAST = ["Okafor", "Bennett", "Ramos", "Mensah", "Chen", "Diallo", "Park", "Silva", "Adeyemi", "Novak", "Haddad", "Brooks",
        "Kamara", "Ortiz", "Nguyen", "Sow", "Patel", "Grant", "Ibrahim", "Moreau", "Tanaka", "Owusu", "Reyes", "Hale",
        "Toure", "Kim", "Duarte", "Asante", "Lindqvist", "Banda", "Coleman", "Fofana", "Ivanova", "Mbeki", "Quinn"]

def names(rng, n):
    return [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]

def bundle_dir(bid):
    d = os.path.join(ROOT, bid)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    return d

def register(bid, meta, files):
    """files: list of (relative path, description). Sizes are measured; a zip of the folder is created."""
    d = os.path.join(ROOT, bid)
    entries = []
    for rel, desc in files:
        p = os.path.join(d, rel)
        assert os.path.exists(p), (bid, rel)
        entries.append(dict(path=rel, desc=desc, kb=round(os.path.getsize(p) / 1024, 1)))
    zpath = os.path.join(ROOT, f"{bid}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, fs in os.walk(d):
            for f in sorted(fs):
                full = os.path.join(base, f)
                z.write(full, os.path.join(bid, os.path.relpath(full, d)))
    total = sum(os.path.getsize(os.path.join(b, f)) for b, _, fs in os.walk(d) for f in fs)
    nfiles = sum(len(fs) for _, _, fs in os.walk(d))
    CATALOG.append(dict(id=bid, files=entries, zip_kb=round(os.path.getsize(zpath) / 1024, 1),
                        total_kb=round(total / 1024, 1), n_files=nfiles, **meta))
    # bundle README for people browsing the repository
    md = [f"# {meta['name']}\n", meta["summary"] + "\n", meta["description"] + "\n",
          f"**Syllabus weeks:** {', '.join(str(w) for w in meta['weeks'])}  \n**Formats:** {', '.join(meta['formats'])}  \n**Level:** {meta['level']}\n",
          "## Files\n", "| File | Size | Contents |", "|---|---|---|"]
    md += [f"| `{e['path']}` | {e['kb']:,} KB | {e['desc']} |" for e in entries]
    md += ["\n## Structure\n"]
    for s in meta["schema"]:
        md += [f"### {s['name']}\n", s.get("note", "") + "\n" if s.get("note") else "", "| Field | Type | Description |", "|---|---|---|"]
        md += [f"| `{c[0]}` | {c[1]} | {c[2]} |" for c in s["fields"]]
        md += [""]
    md += ["## Tasks\n"] + [f"- **Week {t[0]}:** {t[1]}" for t in meta["tasks"]]
    md += ["\n## Starter code\n", "```python", meta["starter"].replace("{BASE}", f"https://YOUR-SITE/wrangling/{bid}/"), "```\n"]
    md += ["<details><summary>Planted challenges (spoilers)</summary>\n"] + [f"- {c}" for c in meta["challenges"]] + ["\n</details>\n",
           "All records are synthetic and for teaching only."]
    open(os.path.join(d, "README.md"), "w").write("\n".join(md))
    print(f"{bid:32s} {nfiles:3d} files  {total/1024/1024:6.2f} MB  zip {os.path.getsize(zpath)/1024/1024:.2f} MB")

# ======================================================================
# W1. Campus Events API: paginated, nested JSON
# ======================================================================
def events_api():
    bid = "campus-events-api"; d = bundle_dir(bid); rng = np.random.default_rng(3101)
    os.makedirs(os.path.join(d, "api", "events"))
    orgs = [{"org_id": f"ORG-{i:03d}", "name": n, "type": t} for i, (n, t) in enumerate([
        ("Data Science Society", "Academic"), ("Black Engineers Collective", "Professional"), ("Chess Club", "Recreation"),
        ("Robotics Team", "Academic"), ("Gospel Choir", "Arts"), ("Entrepreneurs Network", "Professional"),
        ("Film Society", "Arts"), ("Hiking & Outdoors", "Recreation"), ("Math Circle", "Academic"), ("Poetry Collective", "Arts"),
        ("African Students Association", "Cultural"), ("Caribbean Students Association", "Cultural"), ("Esports League", "Recreation"),
        ("Pre-Med Society", "Professional"), ("Debate Union", "Academic"), ("Women in Computing", "Professional")], start=1)]
    venues = [("Blackburn Center Ballroom", "Blackburn Center", "Ballroom", 400), ("Founders Library Room 2", "Founders Library", "2", 60),
              ("Cramton Auditorium", "Cramton", "Main", 1500), ("Douglass Hall 116", "Douglass Hall", "116", 90),
              ("Engineering Lab 3", "LKD Engineering", "L3", 40), ("The Yard", "Outdoor", None, 2000), ("Math Building 201", "Math Building", "201", 70)]
    cats = ["Workshop", "Talk", "Social", "Competition", "Performance", "Career Fair", "Hackathon", "Meeting"]
    tags_pool = ["python", "career", "free food", "networking", "music", "beginner-friendly", "AI", "outdoors", "culture", "games", "health", "finance"]
    events, clean_rows = [], []
    n = 580; students = [f"@{rng.integers(10**7, 10**8)}" for _ in range(3000)]
    for i in range(n):
        org = orgs[rng.integers(len(orgs))]
        start = pd.Timestamp("2025-08-25 09:00", tz="America/New_York") + pd.Timedelta(hours=int(rng.integers(0, 24 * 110)))
        start = start.replace(minute=int(rng.choice([0, 30])))
        online = rng.random() < .12
        v = venues[rng.integers(len(venues))]
        cap = v[3]; nreg = int(min(cap, rng.gamma(2, 12)))
        regs = []
        for s in rng.choice(students, nreg, replace=False):
            rt = start - pd.Timedelta(hours=float(rng.uniform(2, 24 * 14)))
            regs.append({"student_id": s, "registered_at": rt.isoformat(timespec="seconds"), "checked_in": bool(rng.random() < .7)})
        price = float(rng.choice([0, 0, 0, 3, 5, 10, 15]))
        ev = {"id": f"EV-{1001+i}", "title": f"{rng.choice(['Intro to', 'Advanced', 'Night of', 'Spring', 'Fall', 'Annual'])} {rng.choice(['Python', 'Jazz', 'Chess', 'Robotics', 'Poetry', 'Startups', 'Data Viz', 'Film', 'Esports', 'Career Prep'])}",
              "category": str(rng.choice(cats)), "starts_at": start.isoformat(), "duration_minutes": int(rng.choice([60, 90, 120, 180])),
              "status": str(rng.choice(["scheduled", "scheduled", "scheduled", "cancelled", "completed"])),
              "venue": None if online else {"name": v[0], "building": v[1], "room": v[2], "capacity": cap},
              "online_url": f"https://meet.example.edu/{rng.integers(10**6, 10**7)}" if online else None,
              "organizer": {"org_id": org["org_id"], "name": org["name"], "contact": {"email": f"{org['name'].split()[0].lower()}@clubs.example.edu", "phone": None}},
              "tags": sorted(set(rng.choice(tags_pool, int(rng.integers(0, 4)), replace=False).tolist())),
              "ticket": {"price": price, "currency": "USD"}, "registrations": regs}
        clean_rows.append(dict(event_id=ev["id"], title=ev["title"], category=ev["category"], starts_at_utc=start.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),
                               status=ev["status"], is_online=online, venue_name=None if online else v[0], building=None if online else v[1],
                               capacity=None if online else cap, org_id=org["org_id"], org_name=org["name"], n_tags=len(ev["tags"]),
                               ticket_price=price, n_registrations=nreg, n_checked_in=sum(r["checked_in"] for r in regs)))
        # mess: some prices as strings, some timestamps as epoch milliseconds, some missing tags key
        if rng.random() < .15: ev["ticket"]["price"] = f"{price:.2f}"
        if rng.random() < .1: ev["starts_at"] = int(start.timestamp() * 1000)
        if rng.random() < .08: del ev["tags"]
        events.append(ev)
    per = 50; pages = int(np.ceil(n / per))
    for p in range(pages):
        chunk = events[p * per:(p + 1) * per]
        if p in (3, 7):  # pagination overlap: last 3 events of the previous page repeated
            chunk = events[p * per - 3:p * per] + chunk
        body = {"meta": {"page": p + 1, "per_page": per, "total": n, "pages": pages,
                         "next": f"page-{p+2}.json" if p + 1 < pages else None, "generated_at": "2025-12-15T08:00:00Z"},
                "data": chunk}
        json.dump(body, open(os.path.join(d, "api", "events", f"page-{p+1}.json"), "w"), indent=1)
    json.dump({"organizations": orgs}, open(os.path.join(d, "api", "organizations.json"), "w"), indent=1)
    json.dump({"name": "Campus Events API (static mirror)", "version": "1.2",
               "endpoints": {"events": "events/page-1.json (follow meta.next)", "organizations": "organizations.json"},
               "notes": ["Times are ISO 8601 with offset; some legacy records use epoch milliseconds.",
                         "Pages are generated nightly; overlapping records between pages are possible."]},
              open(os.path.join(d, "api", "index.json"), "w"), indent=1)
    os.makedirs(os.path.join(d, "reference"))
    pd.DataFrame(clean_rows).to_csv(os.path.join(d, "reference", "events_flat_clean.csv"), index=False)
    register(bid, dict(
        name="Campus Events API", summary="A paginated JSON API of 580 campus events with nested venues, organizers and registrations.",
        description="A static mirror of a fictional university events API. Each page returns 50 events with nested objects (venue, organizer, contact) "
                    "and nested lists (tags, registrations). Because it is hosted on the website, you can call it with requests exactly like a real API: "
                    "start at page 1 and follow meta.next until it is null.",
        weeks=[3, 4, 14], formats=["JSON", "REST-style API"], level="Intermediate",
        topics=["API retrieval", "Pagination", "Nested JSON", "json_normalize"],
        schema=[dict(name="api/events/page-N.json", note="Top level: {meta, data}. One object in data:", fields=[
            ("id", "string", "Event ID, e.g. EV-1001"), ("title", "string", "Event title"), ("category", "string", "Workshop, Talk, Social ..."),
            ("starts_at", "string or integer", "ISO 8601 with offset, or epoch milliseconds (legacy)"), ("duration_minutes", "integer", "Length"),
            ("status", "string", "scheduled, cancelled or completed"), ("venue", "object or null", "{name, building, room, capacity}; null when online"),
            ("online_url", "string or null", "Link for online events"), ("organizer", "object", "{org_id, name, contact: {email, phone}}"),
            ("tags", "list of strings", "Labels; key missing in some records"), ("ticket", "object", "{price (number or string), currency}"),
            ("registrations", "list of objects", "{student_id, registered_at, checked_in}")]),
            dict(name="api/organizations.json", fields=[("org_id", "string", "Organization ID"), ("name", "string", "Name"), ("type", "string", "Academic, Arts, Cultural ...")])],
        tasks=[(3, "Download every page by following meta.next; stop when it is null"),
               (4, "Join events to organizations.json and count events per organization type"),
               (14, "Flatten with pd.json_normalize into an events table and a registrations table (one row per registration)"),
               (14, "Compute attendance rate (checked in ÷ registered) by category and venue")],
        starter="""import requests, pandas as pd

BASE = "{BASE}api/events/"
url, pages = BASE + "page-1.json", []
while url:
    body = requests.get(url, timeout=30).json()
    pages.extend(body["data"])
    url = BASE + body["meta"]["next"] if body["meta"]["next"] else None

events = pd.json_normalize(pages, sep="_")
regs = pd.json_normalize(pages, record_path="registrations", meta=["id"])
print(events.shape, regs.shape)""",
        challenges=["Pages 4 and 8 repeat the last 3 events of the previous page (deduplicate by id)",
                    "starts_at is an ISO string with offset in most records but epoch milliseconds in about 10%",
                    "ticket.price is a number in most records and a string ('5.00') in about 15%",
                    "The tags key is missing entirely in about 8% of records (not an empty list)",
                    "venue is null for online events, so flattening creates NaN columns that are legitimately empty"],
        reference="reference/events_flat_clean.csv: one row per unique event with times converted to UTC"),
        [("api/index.json", "API description and endpoints"), ("api/events/page-1.json", "First page (12 pages in total, page-1 … page-12)"),
         ("api/organizations.json", "Lookup list of 16 student organizations"), ("reference/events_flat_clean.csv", "Reference: flattened, deduplicated events")])

# ======================================================================
# W2. Riverbend Books: HTML pages to scrape
# ======================================================================
def books_web():
    bid = "riverbend-books-web"; d = bundle_dir(bid); rng = np.random.default_rng(3202)
    os.makedirs(os.path.join(d, "site", "catalog")); os.makedirs(os.path.join(d, "reference"))
    genres = ["Mystery", "Science Fiction", "Romance", "History", "Data & Computing", "Poetry", "Biography", "Young Adult"]
    adj = ["Silent", "Hidden", "Last", "Golden", "Broken", "Electric", "Distant", "Midnight", "Paper", "Crimson", "Quiet", "Infinite"]
    noun = ["River", "Algorithm", "Garden", "Empire", "Signal", "Harbor", "Letter", "Kingdom", "Theorem", "Archive", "Season", "Voyage"]
    books = []
    for i in range(240):
        a = names(rng, 1)[0]
        price = round(float(rng.uniform(7, 45)), 2)
        sale = rng.random() < .15
        books.append(dict(isbn=f"978{rng.integers(10**9, 10**10)}", title=f"The {rng.choice(adj)} {rng.choice(noun)}", author=a,
                          genre=str(rng.choice(genres)), price=round(price * .7, 2) if sale else price, list_price=price,
                          on_sale=bool(sale), rating=(None if rng.random() < .1 else round(float(np.clip(rng.normal(4.0, .5), 1, 5)), 1)),
                          reviews=int(rng.negative_binomial(1, .01)), in_stock=bool(rng.random() > .12), pages=int(rng.integers(120, 720))))
    css = "<style>body{font-family:Georgia,serif;max-width:900px;margin:auto;padding:1rem}.book{border:1px solid #ccc;padding:.6rem;margin:.5rem 0}.price{font-weight:bold}nav a{margin-right:1rem}</style>"
    def stars(r): return "No ratings yet" if r is None else "★" * int(round(r)) + "☆" * (5 - int(round(r))) + f" ({r})"
    def price_txt(b):
        f = rng.random()
        if b["on_sale"]: return f"Sale: ${b['price']:.2f} (was ${b['list_price']:.2f})"
        return f"${b['price']:.2f}" if f < .8 else f"{b['price']:.2f} USD"
    def author_txt(a):
        first, last = a.split(" ", 1)
        return f"by {a}" if rng.random() < .7 else f"{last}, {first}"
    per = 30; pages = 8
    for p in range(pages):
        cards = []
        for b in books[p * per:(p + 1) * per]:
            stock = "In stock" if b["in_stock"] else rng.choice(["Out of stock", "Currently unavailable"])
            cards.append(f"""  <div class="book" data-isbn="{b['isbn']}">
    <h3 class="title"> {b['title']} </h3>
    <span class="author">{author_txt(b['author'])}</span>
    <div class="meta"><span class="genre">{b['genre']}</span> · <span class="pages">{b['pages']} pages</span></div>
    <p class="price">{price_txt(b)}</p>
    <p class="rating">{stars(b['rating'])} · {b['reviews']:,} reviews</p>
    <p class="stock">{stock}</p>
  </div>""")
        prev_l = f'<a href="page-{p}.html" rel="prev">« Previous</a>' if p > 0 else ""
        next_l = f'<a href="page-{p+2}.html" rel="next">Next »</a>' if p + 1 < pages else ""
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Catalog page {p+1} · Riverbend Books</title>{css}</head>
<body><header><a href="../index.html">Riverbend Books</a> › Catalog › Page {p+1} of {pages}</header>
<main id="catalog">
{chr(10).join(cards)}
</main>
<nav class="pagination">{prev_l} <span>Page {p+1}</span> {next_l}</nav>
<footer>Riverbend Books is a fictional store created for teaching web scraping.</footer></body></html>"""
        open(os.path.join(d, "site", "catalog", f"page-{p+1}.html"), "w").write(html)
    top = sorted(books, key=lambda b: -b["reviews"])[:25]
    rows = "\n".join(f"<tr><td>{k+1}</td><td>{b['title']}</td><td>{b['author']}</td><td>{b['reviews']:,}</td><td>${b['price']:.2f}</td></tr>" for k, b in enumerate(top))
    open(os.path.join(d, "site", "bestsellers.html"), "w").write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Bestsellers · Riverbend Books</title>{css}</head>
<body><header><a href="index.html">Riverbend Books</a> › Bestsellers</header><h1>Top 25 by number of reviews</h1>
<table id="bestsellers" border="1"><thead><tr><th>Rank</th><th>Title</th><th>Author</th><th>Reviews</th><th>Price</th></tr></thead><tbody>
{rows}
</tbody></table></body></html>""")
    open(os.path.join(d, "site", "index.html"), "w").write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Riverbend Books</title>{css}</head>
<body><h1>Riverbend Books</h1><p>An independent (and fictional) bookstore.</p>
<ul><li><a href="catalog/page-1.html">Browse the catalog</a> ({len(books)} titles)</li><li><a href="bestsellers.html">Bestsellers</a></li></ul></body></html>""")
    open(os.path.join(d, "site", "robots.txt"), "w").write("User-agent: *\nAllow: /\nCrawl-delay: 1\n")
    pd.DataFrame(books).to_csv(os.path.join(d, "reference", "books_clean.csv"), index=False)
    register(bid, dict(
        name="Riverbend Books Web Store", summary="Eight catalog pages and a bestseller table to scrape from a fictional bookstore website.",
        description="A small, static bookstore website hosted alongside the library, so you can practice real web scraping with requests and "
                    "BeautifulSoup without breaking anyone's terms of service. Book cards are spread across 8 paginated catalog pages, and a bestseller "
                    "page holds an HTML table you can read with pd.read_html.",
        weeks=[3, 5], formats=["HTML", "Web scraping"], level="Beginner",
        topics=["Web scraping", "BeautifulSoup", "Pagination", "Regular expressions", "pd.read_html"],
        schema=[dict(name="site/catalog/page-N.html", note="Each book is a <div class=\"book\" data-isbn=\"…\"> containing:", fields=[
            ("h3.title", "text", "Title (with stray spaces)"), ("span.author", "text", "'by First Last' or 'Last, First'"),
            ("span.genre / span.pages", "text", "Genre and page count ('352 pages')"),
            ("p.price", "text", "'$12.99', '12.99 USD' or 'Sale: $9.09 (was $12.99)'"),
            ("p.rating", "text", "'★★★★☆ (4.1) · 1,234 reviews' or 'No ratings yet'"), ("p.stock", "text", "In stock / Out of stock / Currently unavailable"),
            ("nav a[rel=next]", "link", "Next page; absent on the last page")]),
            dict(name="site/bestsellers.html", fields=[("table#bestsellers", "HTML table", "Rank, Title, Author, Reviews, Price")])],
        tasks=[(3, "Scrape all 240 books by following the Next links; respect robots.txt and wait 1 second between requests"),
               (3, "Read the bestseller table with pd.read_html"),
               (5, "Use regular expressions to split price into current and list price, and rating into stars and review count"),
               (5, "Standardize author names to 'First Last' and compare your table with reference/books_clean.csv")],
        starter="""import time, requests, pandas as pd
from bs4 import BeautifulSoup

BASE = "{BASE}site/catalog/"
url, rows = BASE + "page-1.html", []
while url:
    soup = BeautifulSoup(requests.get(url, timeout=30).text, "html.parser")
    for card in soup.select("div.book"):
        rows.append({"isbn": card["data-isbn"],
                     "title": card.select_one(".title").get_text(strip=True),
                     "author": card.select_one(".author").get_text(strip=True),
                     "price_text": card.select_one(".price").get_text(strip=True)})
    nxt = soup.select_one("nav a[rel=next]")
    url = BASE + nxt["href"] if nxt else None
    time.sleep(1)   # be polite (robots.txt asks for a 1-second delay)

books = pd.DataFrame(rows)
print(books.shape)""",
        challenges=["Prices appear in three text formats, including sale prices that contain two amounts",
                    "Authors are written 'by First Last' or 'Last, First'", "Titles have leading and trailing spaces",
                    "Ratings are star characters plus a number; unrated books say 'No ratings yet'",
                    "Out-of-stock books use two different phrases", "The last page has no Next link: your loop must stop cleanly"],
        reference="reference/books_clean.csv: the true values for all 240 books"),
        [("site/index.html", "Store home page"), ("site/catalog/page-1.html", "First of 8 catalog pages (30 books each)"),
         ("site/bestsellers.html", "Bestseller table (top 25)"), ("site/robots.txt", "Crawling rules"),
         ("reference/books_clean.csv", "Reference: all 240 books as clean data")])

# ======================================================================
# W3. Northstar Retail: integrate CSV, JSON, Excel, TSV
# ======================================================================
def retail():
    bid = "northstar-retail-integration"; d = bundle_dir(bid); rng = np.random.default_rng(3303)
    os.makedirs(os.path.join(d, "reference"))
    cities = ["Riverton", "Harbor City", "Maple Valley", "Old Port", "Cedar Park", "Summit"]
    nc = 2000
    cust = pd.DataFrame({"cid": np.arange(1, nc + 1), "name": names(rng, nc), "city": rng.choice(cities, nc),
                         "signup": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1000, nc), unit="D"),
                         "segment": rng.choice(["Consumer", "Small Business", "Corporate"], nc, p=[.7, .2, .1])})
    cust["email"] = [f"{n.split()[0].lower()}.{n.split()[1].lower()}{i}@mail.example.com" for i, n in zip(cust.cid, cust.name)]
    cats = {"Electronics": (40, 400), "Home": (10, 120), "Outdoor": (15, 250), "Office": (3, 60), "Toys": (8, 80)}
    prods = []
    for i in range(1, 181):
        c = rng.choice(list(cats)); lo, hi = cats[c]; cost = round(float(rng.uniform(lo, hi)), 2)
        prods.append(dict(sku=f"NS-{1000+i}", product_name=f"{rng.choice(['Pro', 'Lite', 'Max', 'Mini', 'Smart', 'Eco'])} {rng.choice(['Lamp', 'Speaker', 'Tent', 'Chair', 'Drone', 'Kettle', 'Backpack', 'Monitor', 'Puzzle', 'Notebook'])} {i}",
                          category=c, unit_cost=cost, list_price=round(cost * rng.uniform(1.3, 1.9), 2), discontinued=bool(rng.random() < .08)))
    prods = pd.DataFrame(prods)
    stores = pd.DataFrame({"store_code": ["RVT-01", "RVT-02", "HBC-01", "MPV-01", "OPT-01", "CDP-01", "WEB"],
                           "store_name": ["Riverton Downtown", "Riverton Mall", "Harbor City Pier", "Maple Valley Center", "Old Port Square", "Cedar Park Plaza", "Online store"],
                           "channel": ["Store"] * 6 + ["Online"], "opened": ["2015-03-01", "2018-09-15", "2016-05-20", "2019-11-01", "2021-04-10", "2022-08-01", "2017-01-01"]})
    no = 8000; orders, lines = [], []
    active = prods[~prods.discontinued]
    for k in range(no):
        cid = int(rng.integers(1, nc + 1)); ts = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 640)))
        st = str(rng.choice(stores.store_code, p=[.15, .12, .14, .1, .1, .09, .3]))
        items = []
        for p in active.sample(int(rng.choice([1, 1, 2, 2, 3, 4])), random_state=int(rng.integers(1e9))).itertuples():
            q = int(rng.choice([1, 1, 1, 2, 3])); price = round(p.list_price * rng.choice([1, 1, 1, .9, .8]), 2)
            items.append({"sku": p.sku, "qty": q, "unit_price": price})
            lines.append(dict(order_id=f"SO-{50000+k}", order_ts=ts.strftime("%Y-%m-%d %H:%M:%S"), customer_id=cid, store_code=st,
                              sku=p.sku, category=p.category, qty=q, unit_price=price, line_revenue=round(q * price, 2),
                              line_cost=round(q * p.unit_cost, 2)))
        orders.append({"order_id": f"SO-{50000+k}", "order_ts": ts.isoformat(), "customer": cid, "store": st,
                       "payment": {"method": str(rng.choice(["card", "cash", "gift card", "mobile"])), "installments": int(rng.choice([1, 1, 1, 3]))},
                       "items": items})
    # ----- write raw sources with integration problems
    raw_c = cust.copy()
    raw_c["customer_id"] = [f"C{c:05d}" for c in raw_c.cid]
    raw_c["signup_date"] = raw_c.signup.dt.strftime("%m/%d/%Y")
    dup = raw_c.sample(40, random_state=1).copy(); dup["email"] = dup.email.str.upper(); dup["customer_id"] = [f"C{c:05d}" for c in range(nc + 1, nc + 41)]
    raw_c = pd.concat([raw_c, dup]).sample(frac=1, random_state=2)
    raw_c[["customer_id", "name", "email", "city", "segment", "signup_date"]].to_csv(os.path.join(d, "customers.csv"), index=False)
    for o in orders[:25]: o["customer"] = int(rng.integers(nc + 100, nc + 400))  # orphans: customers that do not exist
    for o in orders:
        if rng.random() < .1:
            for it in o["items"]: it["sku"] = it["sku"].lower() if rng.random() < .5 else " " + it["sku"] + " "
    json.dump(orders, open(os.path.join(d, "orders.json"), "w"), indent=1)
    with pd.ExcelWriter(os.path.join(d, "products.xlsx")) as xw:
        prods[~prods.discontinued].drop(columns="discontinued").to_excel(xw, sheet_name="Products", index=False)
        prods[prods.discontinued].drop(columns="discontinued").to_excel(xw, sheet_name="Discontinued", index=False)
    stores.to_csv(os.path.join(d, "stores.tsv"), sep="\t", index=False)
    L = pd.DataFrame(lines)
    ret = L.sample(600, random_state=4)[["order_id", "sku", "qty", "unit_price"]].copy()
    ret["reason"] = rng.choice(["Damaged", "Wrong item", "Changed mind", "Defective", "damaged"], len(ret))
    ret["refund_amount"] = [f"${q*p:,.2f}" for q, p in zip(ret.qty, ret.unit_price)]
    ret[["order_id", "sku", "reason", "refund_amount"]].to_csv(os.path.join(d, "returns.csv"), index=False)
    # ----- reference
    L = L[~L.order_id.isin([o["order_id"] for o in orders[:25]])]
    L = L.merge(cust[["cid", "city", "segment"]], left_on="customer_id", right_on="cid").drop(columns="cid")
    L["returned"] = L.set_index(["order_id", "sku"]).index.isin(ret.set_index(["order_id", "sku"]).index)
    L.to_csv(os.path.join(d, "reference", "order_lines_clean.csv"), index=False)
    register(bid, dict(
        name="Northstar Retail Integration", summary="Customers, orders, products, stores and returns spread across CSV, JSON, Excel and TSV files.",
        description="Five source files from a fictional retail chain, each exported by a different system. Your job is to integrate them into "
                    "one analysis-ready table of order lines (one row per product sold). The keys do not match cleanly across systems, which is the point.",
        weeks=[4, 5, 6], formats=["CSV", "JSON", "Excel", "TSV"], level="Intermediate",
        topics=["Merging", "Key matching", "Excel sheets", "Nested JSON", "Deduplication"],
        schema=[dict(name="customers.csv", fields=[("customer_id", "string", "'C00012' format"), ("name", "string", "Full name"), ("email", "string", "Email"),
                                                   ("city", "string", "City"), ("segment", "string", "Consumer, Small Business or Corporate"), ("signup_date", "string", "MM/DD/YYYY")]),
                dict(name="orders.json", note="A list of orders:", fields=[("order_id", "string", "SO-50000 …"), ("order_ts", "string", "ISO 8601"),
                     ("customer", "integer", "Customer number (12, not 'C00012')"), ("store", "string", "Store code"),
                     ("payment", "object", "{method, installments}"), ("items", "list", "{sku, qty, unit_price}")]),
                dict(name="products.xlsx", note="Two sheets: Products and Discontinued", fields=[("sku", "string", "NS-1001 …"), ("product_name", "string", "Name"),
                     ("category", "string", "Category"), ("unit_cost", "number", "Cost"), ("list_price", "number", "Regular price")]),
                dict(name="stores.tsv", fields=[("store_code", "string", "e.g. RVT-01, WEB"), ("store_name", "string", "Name"), ("channel", "string", "Store or Online"), ("opened", "date", "Opening date")]),
                dict(name="returns.csv", fields=[("order_id", "string", "Order"), ("sku", "string", "Product"), ("reason", "string", "Return reason"), ("refund_amount", "string", "'$1,234.56'")])],
        tasks=[(4, "Explode orders.json into one row per item and merge with customers, products and stores"),
               (4, "Make the keys match: customer number vs 'C00012', SKU case and spaces"),
               (5, "Find the duplicate customers and the orders whose customer does not exist"),
               (6, "Add features: line revenue, margin, returned flag, customer tenure; compare with reference/order_lines_clean.csv")],
        starter="""import json, requests, pandas as pd

BASE = "{BASE}"
customers = pd.read_csv(BASE + "customers.csv")
orders = pd.json_normalize(requests.get(BASE + "orders.json").json(),
                           record_path="items", meta=["order_id", "order_ts", "customer", "store"])
products = pd.read_excel(BASE + "products.xlsx", sheet_name=None)   # dict of both sheets
stores = pd.read_csv(BASE + "stores.tsv", sep="\\t")
returns = pd.read_csv(BASE + "returns.csv")
print(orders.head())""",
        challenges=["orders.json uses a customer number (12) while customers.csv uses 'C00012'",
                    "About 10% of orders have SKUs in lowercase or with surrounding spaces",
                    "40 customers appear twice with the same email in different case and a new customer_id",
                    "25 orders reference customers that do not exist (orphans)",
                    "Products are split across two Excel sheets; discontinued products never appear in orders",
                    "Return reasons mix 'Damaged' and 'damaged'; refund amounts are text with '$' and commas"],
        reference="reference/order_lines_clean.csv: the integrated table (orphans removed, keys fixed, returned flag added)"),
        [("customers.csv", "2,040 customer records (CSV)"), ("orders.json", "8,000 orders with nested items (JSON)"),
         ("products.xlsx", "180 products on two sheets (Excel)"), ("stores.tsv", "7 stores (tab-separated)"), ("returns.csv", "600 returned items (CSV)"),
         ("reference/order_lines_clean.csv", "Reference: integrated order lines")])

# ======================================================================
# W4. Riverton General Hospital: relational database (SQLite)
# ======================================================================
def hospital_db():
    bid = "riverton-hospital-db"; d = bundle_dir(bid); rng = np.random.default_rng(3404)
    os.makedirs(os.path.join(d, "csv"))
    depts = pd.DataFrame({"dept_id": range(1, 11), "dept_name": ["Emergency", "Cardiology", "Oncology", "Pediatrics", "Orthopedics", "Neurology",
                          "General Surgery", "Internal Medicine", "Obstetrics", "Psychiatry"], "floor": [1, 3, 4, 2, 5, 6, 5, 3, 2, 7],
                          "beds": [40, 32, 28, 30, 26, 22, 34, 48, 24, 20]})
    nd = 80
    docs = pd.DataFrame({"doctor_id": range(101, 101 + nd), "full_name": ["Dr. " + x for x in names(rng, nd)],
                         "dept_id": rng.integers(1, 11, nd), "hire_date": (pd.Timestamp("2005-01-01") + pd.to_timedelta(rng.integers(0, 7000, nd), unit="D")).strftime("%Y-%m-%d"),
                         "specialty_level": rng.choice(["Attending", "Fellow", "Resident"], nd, p=[.5, .2, .3])})
    np_ = 5000
    pts = pd.DataFrame({"patient_id": range(1, np_ + 1), "full_name": names(rng, np_),
                        "birth_date": (pd.Timestamp("1935-01-01") + pd.to_timedelta(rng.integers(0, 32000, np_), unit="D")).strftime("%Y-%m-%d"),
                        "sex": rng.choice(["F", "M"], np_), "zip_code": rng.choice(["20001", "20002", "20009", "20010", "20011", "20017", "20019", "20032"], np_),
                        "insurance": rng.choice(["Medicaid", "Medicare", "Private", "Uninsured"], np_, p=[.3, .25, .38, .07])})
    dup = pts.sample(35, random_state=3).copy(); dup["patient_id"] = range(np_ + 1, np_ + 36)
    pts = pd.concat([pts, dup])
    codes = [("I10", "Hypertension"), ("E11", "Type 2 diabetes"), ("J18", "Pneumonia"), ("I21", "Heart attack"), ("N39", "Urinary tract infection"),
             ("S72", "Hip fracture"), ("C34", "Lung cancer"), ("O80", "Normal delivery"), ("F32", "Depression"), ("G40", "Epilepsy"),
             ("K35", "Appendicitis"), ("J45", "Asthma"), ("I50", "Heart failure"), ("A41", "Sepsis"), ("R07", "Chest pain"), ("M54", "Back pain")]
    diag = pd.DataFrame(codes, columns=["code", "description"])
    na = 12000
    adm_dept = rng.integers(1, 11, na)
    admit = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.uniform(0, 1000, na), unit="D")
    los = np.round(rng.gamma(1.6, 2.6, na), 2)
    dis = admit + pd.to_timedelta(los, unit="D")
    adm = pd.DataFrame({"admission_id": range(1, na + 1), "patient_id": rng.integers(1, np_ + 1, na), "dept_id": adm_dept,
                        "attending_id": [int(rng.choice(docs.doctor_id[docs.dept_id == dd])) if (docs.dept_id == dd).any() else 101 for dd in adm_dept],
                        "admit_time": admit.floor("min").strftime("%Y-%m-%d %H:%M"), "discharge_time": dis.floor("min").strftime("%Y-%m-%d %H:%M"),
                        "admission_type": rng.choice(["Emergency", "Elective", "Urgent", "Transfer"], na, p=[.5, .25, .2, .05]),
                        "discharge_status": rng.choice(["Home", "Home with care", "Transferred", "Deceased"], na, p=[.72, .17, .08, .03])})
    still = rng.choice(na, 60, replace=False); adm.loc[still, ["discharge_time", "discharge_status"]] = None
    bad = rng.choice(na, 20, replace=False)
    adm.loc[bad, "discharge_time"] = (admit[bad] - pd.Timedelta(hours=5)).floor("min").strftime("%Y-%m-%d %H:%M")
    adm.loc[rng.choice(na, 15, replace=False), "attending_id"] = 999  # doctor that does not exist
    ad = []
    for a in adm.admission_id:
        for k, c in enumerate(rng.choice(diag.code, int(rng.choice([1, 1, 2, 3])), replace=False)):
            ad.append((int(a), str(c), k == 0))
    adx = pd.DataFrame(ad, columns=["admission_id", "code", "is_primary"])
    tests = [("Hemoglobin", "g/dL", 13.5, 1.5), ("Glucose", "mg/dL", 105, 25), ("Creatinine", "mg/dL", 1.0, .3),
             ("Sodium", "mmol/L", 139, 3), ("Potassium", "mmol/L", 4.2, .45), ("WBC", "10^3/uL", 8, 2.5)]
    labs = []
    for a in adm.itertuples():
        for _ in range(int(rng.integers(1, 7))):
            t = tests[rng.integers(len(tests))]
            v = round(float(rng.normal(t[2], t[3])), 2)
            labs.append((int(a.admission_id), t[0], v, t[1], a.admit_time))
    labs = pd.DataFrame(labs, columns=["admission_id", "test_name", "value", "unit", "collected_at"])
    labs["lab_id"] = range(1, len(labs) + 1)
    labs.loc[labs.sample(25, random_state=6).index, "value"] = -1
    labs = labs[["lab_id", "admission_id", "test_name", "value", "unit", "collected_at"]]
    bill = pd.DataFrame({"admission_id": adm.admission_id, "total_charge": np.round(los * rng.uniform(1800, 4200, na) + 900, 2),
                         "insurance_paid": 0.0, "patient_paid": 0.0})
    bill["insurance_paid"] = np.round(bill.total_charge * rng.uniform(.5, .95, na), 2)
    bill["patient_paid"] = np.round((bill.total_charge - bill.insurance_paid) * rng.uniform(0, 1, na), 2)
    tables = {"departments": depts, "doctors": docs, "patients": pts, "diagnosis_codes": diag, "admissions": adm,
              "admission_diagnoses": adx, "lab_results": labs, "billing": bill}
    ddl = """-- Riverton General Hospital (fictional). Foreign keys are declared but NOT enforced, as in many legacy systems.
CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, dept_name TEXT NOT NULL, floor INTEGER, beds INTEGER);
CREATE TABLE doctors (doctor_id INTEGER PRIMARY KEY, full_name TEXT, dept_id INTEGER REFERENCES departments(dept_id), hire_date TEXT, specialty_level TEXT);
CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, full_name TEXT, birth_date TEXT, sex TEXT, zip_code TEXT, insurance TEXT);
CREATE TABLE diagnosis_codes (code TEXT PRIMARY KEY, description TEXT);
CREATE TABLE admissions (admission_id INTEGER PRIMARY KEY, patient_id INTEGER REFERENCES patients(patient_id), dept_id INTEGER REFERENCES departments(dept_id),
  attending_id INTEGER REFERENCES doctors(doctor_id), admit_time TEXT, discharge_time TEXT, admission_type TEXT, discharge_status TEXT);
CREATE TABLE admission_diagnoses (admission_id INTEGER REFERENCES admissions(admission_id), code TEXT REFERENCES diagnosis_codes(code), is_primary INTEGER);
CREATE TABLE lab_results (lab_id INTEGER PRIMARY KEY, admission_id INTEGER REFERENCES admissions(admission_id), test_name TEXT, value REAL, unit TEXT, collected_at TEXT);
CREATE TABLE billing (admission_id INTEGER PRIMARY KEY REFERENCES admissions(admission_id), total_charge REAL, insurance_paid REAL, patient_paid REAL);
"""
    open(os.path.join(d, "schema.sql"), "w").write(ddl)
    db = os.path.join(d, "riverton_hospital.db")
    con = sqlite3.connect(db); con.executescript(ddl)
    for t, df in tables.items():
        df.to_sql(t, con, if_exists="append", index=False)
        df.to_csv(os.path.join(d, "csv", f"{t}.csv"), index=False)
    con.commit(); con.execute("VACUUM"); con.close()
    open(os.path.join(d, "er_diagram.md"), "w").write("""# Entity-relationship diagram

```mermaid
erDiagram
    departments ||--o{ doctors : employs
    departments ||--o{ admissions : receives
    patients ||--o{ admissions : has
    doctors ||--o{ admissions : attends
    admissions ||--o{ admission_diagnoses : has
    diagnosis_codes ||--o{ admission_diagnoses : describes
    admissions ||--o{ lab_results : produces
    admissions ||--|| billing : billed
```
""")
    open(os.path.join(d, "sql_exercises.md"), "w").write("""# SQL exercises (Riverton General Hospital)

## Week 7: basics
1. List the 10 most recent admissions (admission_id, admit_time, admission_type).
2. How many admissions are there per admission_type? Sort from most to least.
3. Which patients are uninsured and born before 1950?
4. What is the average total_charge in billing, rounded to 2 decimals?

## Week 8: joins, subqueries, transformation
5. For each department, count admissions and compute the average length of stay in days (use julianday()).
6. List each attending doctor with their number of patients; include doctors with zero admissions (LEFT JOIN).
7. Find admissions whose attending_id does not match any doctor (anti-join).
8. Which patients have been admitted more than 3 times? Use a subquery or HAVING.
9. Top 5 primary diagnoses by number of admissions, with their descriptions.
10. For each admission, the percentage of the bill paid by insurance; then the average by insurance type.
11. Readmissions: admissions that start within 30 days of the same patient's previous discharge (self-join or window function LAG).
12. Rank departments by average total charge per day of stay (window function RANK).

## Week 9: data quality with SQL
13. Count NULL discharge times. Are they all patients still in hospital?
14. Find admissions discharged before they were admitted.
15. Find likely duplicate patients (same full_name and birth_date, different patient_id).
16. Find lab values that are impossible (negative).
""")
    register(bid, dict(
        name="Riverton General Hospital Database", summary="A relational SQLite database with 8 linked tables: patients, admissions, doctors, labs and billing.",
        description="A fictional hospital's database with 5,000 patients, 12,000 admissions and about 42,000 lab results across 8 related tables. "
                    "Use it to practice SQL from simple SELECTs to joins, subqueries and window functions, and to audit referential integrity. "
                    "Every table is also provided as CSV for pandas merges.",
        weeks=[7, 8, 9], formats=["SQLite", "SQL", "CSV"], level="Intermediate",
        topics=["SQL", "Joins", "Subqueries", "Window functions", "Referential integrity"],
        schema=[dict(name=t, fields=[(c, str(df[c].dtype).replace("object", "text").replace("int64", "integer").replace("float64", "real").replace("bool", "boolean"), "")
                                     for c in df.columns]) for t, df in tables.items()],
        tasks=[(7, "Answer exercises 1-4 in sql_exercises.md"), (8, "Answer exercises 5-12, including a LEFT JOIN, an anti-join and a window function"),
               (9, "Audit the database with exercises 13-16 and write a short data-quality report")],
        starter="""import sqlite3, urllib.request, pandas as pd

urllib.request.urlretrieve("{BASE}riverton_hospital.db", "riverton_hospital.db")
con = sqlite3.connect("riverton_hospital.db")

pd.read_sql(\"\"\"
    SELECT d.dept_name, COUNT(*) AS admissions,
           ROUND(AVG(julianday(a.discharge_time) - julianday(a.admit_time)), 2) AS avg_los_days
    FROM admissions a JOIN departments d ON d.dept_id = a.dept_id
    GROUP BY d.dept_name ORDER BY admissions DESC
\"\"\", con)""",
        challenges=["60 admissions have no discharge time (patients still in hospital): legitimately NULL",
                    "20 admissions are discharged before they were admitted",
                    "15 admissions reference attending_id 999, a doctor who does not exist (foreign keys are not enforced)",
                    "35 patients appear twice with the same name and birth date under a new patient_id",
                    "25 lab results have the impossible value -1"],
        reference="sql_exercises.md lists the questions; answers are left to the instructor"),
        [("riverton_hospital.db", "SQLite database (8 tables)"), ("schema.sql", "Table definitions (DDL)"), ("er_diagram.md", "Entity-relationship diagram"),
         ("sql_exercises.md", "16 graded SQL exercises by week"), ("csv/admissions.csv", "Every table as CSV (8 files in csv/)")])

# ======================================================================
# W5. GridWise smart-meter pipeline: daily drops, schema drift, validation
# ======================================================================
def meter_pipeline():
    bid = "gridwise-meter-pipeline"; d = bundle_dir(bid); rng = np.random.default_rng(3505)
    for sub in ["incoming", "late", "reference", "dags"]: os.makedirs(os.path.join(d, sub))
    nm = 180
    meters = pd.DataFrame({"meter_id": [f"MTR-{10000+i}" for i in range(nm)],
                           "customer_type": rng.choice(["Residential", "Commercial", "Industrial"], nm, p=[.75, .2, .05]),
                           "district": rng.choice(["North", "South", "East", "West", "Downtown"], nm),
                           "install_date": (pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2200, nm), unit="D")).strftime("%Y-%m-%d"),
                           "tariff": rng.choice(["Standard", "Time-of-use", "Solar net metering"], nm, p=[.6, .3, .1])})
    meters.to_csv(os.path.join(d, "reference", "meters.csv"), index=False)
    scale = meters.customer_type.map({"Residential": .9, "Commercial": 6, "Industrial": 30}).values
    days = pd.date_range("2025-06-01", "2025-06-30")
    daily = []
    for day in days:
        h = np.arange(24)
        prof = .5 + .6 * np.exp(-((h - 19) ** 2) / 8) + .35 * np.exp(-((h - 8) ** 2) / 5)
        heat = 1 + .015 * (day.day)  # warming through June
        rows = []
        for m, s in zip(meters.meter_id, scale):
            kwh = np.round(s * prof * heat * rng.lognormal(0, .15, 24), 3)
            for hh in range(24):
                rows.append((m, (day + pd.Timedelta(hours=hh)).strftime("%Y-%m-%d %H:%M"), kwh[hh],
                             round(float(rng.normal(240, 3)), 1)))
        df = pd.DataFrame(rows, columns=["meter_id", "reading_time", "kwh", "voltage"])
        daily.append(df)
        clean_day = df.copy()
        ds = day.strftime("%Y-%m-%d")
        raw = df.copy()
        # injected issues
        i = rng.choice(len(raw), 6, replace=False); raw.loc[i, "kwh"] = -raw.loc[i, "kwh"]
        i = rng.choice(len(raw), 3, replace=False); raw.loc[i, "kwh"] = 9999.0
        if day.day % 5 == 0:
            ghost = raw.sample(24, random_state=day.day).copy(); ghost["meter_id"] = "MTR-99999"; raw = pd.concat([raw, ghost])
        raw = raw.drop(raw.sample(frac=.004, random_state=day.day).index)  # dropped readings
        if day.day < 20: raw = raw.drop(columns="voltage")               # voltage column appears on 2025-06-20
        if day.day >= 15: raw = raw.rename(columns={"kwh": "energy_kwh"})  # renamed on 2025-06-15
        path = os.path.join(d, "incoming", f"readings_{ds}.csv")
        if day.day == 11:
            txt = raw.to_csv(index=False, sep=";", decimal=",")
            open(path, "w").write(txt)
        elif day.day == 23:
            half = len(raw) // 2
            txt = raw.iloc[:half].to_csv(index=False) + raw.iloc[half:].to_csv(index=False)  # header repeated mid-file
            open(path, "w").write(txt)
        elif day.day == 27:
            raw.to_csv(os.path.join(d, "late", f"readings_{ds}.csv"), index=False)
        else:
            raw.to_csv(path, index=False)
        if day.day == 8:
            raw.sample(frac=.5, random_state=8).to_csv(os.path.join(d, "incoming", f"readings_{ds}_resend.csv"), index=False)
    allc = pd.concat(daily)
    allc["date"] = allc.reading_time.str[:10]
    agg = allc.groupby(["date", "meter_id"]).agg(total_kwh=("kwh", "sum"), peak_kwh=("kwh", "max"), readings=("kwh", "size")).round(3).reset_index()
    agg.to_csv(os.path.join(d, "reference", "daily_usage_clean.csv"), index=False)
    open(os.path.join(d, "validation_rules.yaml"), "w").write("""# Validation rules for incoming GridWise meter files
dataset: meter_readings
target_schema:
  meter_id: {type: string, pattern: "^MTR-\\\\d{5}$", must_exist_in: reference/meters.csv}
  reading_time: {type: datetime, format: "%Y-%m-%d %H:%M", hourly: true}
  energy_kwh: {type: float, min: 0, max: 500, aliases: [kwh]}
  voltage: {type: float, min: 200, max: 260, required_from: "2025-06-20"}
rules:
  - id: R1
    description: Each file has exactly one header row and a consistent delimiter (comma).
  - id: R2
    description: energy_kwh must be between 0 and 500; values outside go to a quarantine table.
  - id: R3
    description: meter_id must exist in the meter registry.
  - id: R4
    description: (meter_id, reading_time) is unique across all loaded data; re-sent files must not create duplicates.
  - id: R5
    description: Every meter should have 24 readings per day; report meters with gaps.
  - id: R6
    description: Every calendar day in the period must be loaded exactly once, including files that arrive late.
""")
    open(os.path.join(d, "pipeline_spec.md"), "w").write("""# Pipeline specification

**Goal:** load every daily file into a small warehouse (SQLite is fine) and publish daily usage per meter.

1. **Extract**: read every file in `incoming/` and `late/`. Detect the delimiter and decimal separator.
2. **Validate**: apply the rules in `validation_rules.yaml`. Rows that fail go to a `quarantine` table with the rule ID.
3. **Transform**: standardize column names (`kwh` → `energy_kwh`), parse timestamps, add `date`, join `reference/meters.csv`.
4. **Load**: `dim_meter`, `fact_reading` (unique on meter_id + reading_time) and `agg_daily_usage` (meter, date, total_kwh, peak_kwh, readings).
5. **Idempotency**: running the pipeline twice must not change the results.
6. **Orchestrate**: wrap the steps in an Airflow DAG (see `dags/gridwise_daily.py`) that runs once per day.

Check your `agg_daily_usage` against `reference/daily_usage_clean.csv` (computed before any problems were injected,
so small differences from dropped and quarantined readings are expected; explain them).
""")
    open(os.path.join(d, "dags", "gridwise_daily.py"), "w").write('''"""Skeleton Airflow DAG for the GridWise pipeline. Fill in the TODOs."""
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def extract(ds, **_):
    # TODO: read incoming/readings_{ds}.csv (and late/ if present); detect delimiter
    ...


def validate(ds, **_):
    # TODO: apply validation_rules.yaml; send failures to a quarantine table
    ...


def load(ds, **_):
    # TODO: upsert into fact_reading (unique meter_id + reading_time), refresh agg_daily_usage
    ...


with DAG("gridwise_daily", start_date=datetime(2025, 6, 1), end_date=datetime(2025, 6, 30),
         schedule="@daily", catchup=True, tags=["teaching"]) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="validate", python_callable=validate)
    t3 = PythonOperator(task_id="load", python_callable=load)
    t1 >> t2 >> t3
''')
    files = [(f"incoming/readings_2025-06-01.csv", "First daily file (29 files in incoming/, one per day, plus a re-sent file)"),
             ("incoming/readings_2025-06-11.csv", "A file exported with semicolons and decimal commas"),
             ("late/readings_2025-06-27.csv", "A file that arrived late, in a different folder"),
             ("reference/meters.csv", "Meter registry: 180 meters"), ("validation_rules.yaml", "Validation rules R1-R6"),
             ("pipeline_spec.md", "What the pipeline must produce"), ("dags/gridwise_daily.py", "Airflow DAG skeleton"),
             ("reference/daily_usage_clean.csv", "Reference: daily usage per meter")]
    register(bid, dict(
        name="GridWise Smart-Meter Pipeline", summary="30 days of hourly smart-meter files that arrive daily, with schema drift, bad files and validation rules.",
        description="A fictional utility receives one CSV of hourly electricity readings per day from 180 smart meters. Build an ETL pipeline that "
                    "validates, cleans and loads them into a warehouse. The files change over the month exactly the way real feeds do: a renamed column, "
                    "a new column, a file in a European format, a duplicated header, a re-sent file and a late arrival.",
        weeks=[9, 10, 11, 12], formats=["CSV", "YAML", "Airflow"], level="Advanced",
        topics=["ETL", "Schema drift", "Validation rules", "Idempotent loads", "Airflow"],
        schema=[dict(name="incoming/readings_YYYY-MM-DD.csv", note="Columns change during the month:", fields=[
                    ("meter_id", "string", "MTR-10000 …"), ("reading_time", "string", "YYYY-MM-DD HH:MM, hourly"),
                    ("kwh / energy_kwh", "float", "Energy used in the hour; renamed to energy_kwh from 2025-06-15"),
                    ("voltage", "float", "Average voltage; present only from 2025-06-20")]),
                dict(name="reference/meters.csv", fields=[("meter_id", "string", "Meter"), ("customer_type", "string", "Residential, Commercial or Industrial"),
                     ("district", "string", "District"), ("install_date", "date", "Installed"), ("tariff", "string", "Pricing plan")])],
        tasks=[(9, "Profile all files: row counts, columns, delimiters, ranges and nulls per day"),
               (10, "Implement rules R1-R6 from validation_rules.yaml as automated checks (pytest or plain asserts)"),
               (11, "Build the extract-validate-transform-load pipeline into SQLite, idempotently"),
               (12, "Orchestrate it with the Airflow DAG skeleton and backfill the whole month")],
        starter="""import glob, pandas as pd

# Clone or download the bundle first; pipelines work on local folders
def read_any(path):
    with open(path) as f:
        head = f.readline()
    sep, dec = (";", ",") if head.count(";") > head.count(",") else (",", ".")
    df = pd.read_csv(path, sep=sep, decimal=dec)
    return df.rename(columns={"kwh": "energy_kwh"}).assign(source_file=path)

files = sorted(glob.glob("gridwise-meter-pipeline/incoming/*.csv") + glob.glob("gridwise-meter-pipeline/late/*.csv"))
readings = pd.concat([read_any(f) for f in files], ignore_index=True)
print(readings.shape, readings.columns.tolist())""",
        challenges=["kwh is renamed energy_kwh from 2025-06-15; voltage appears only from 2025-06-20",
                    "2025-06-11 uses ';' as the delimiter and ',' as the decimal separator",
                    "2025-06-23 contains the header row a second time in the middle of the file",
                    "2025-06-27 is missing from incoming/ and arrives in late/",
                    "readings_2025-06-08_resend.csv repeats half of that day's rows",
                    "Every day has 6 negative readings and 3 readings of 9999 (sensor error codes)",
                    "Every fifth day includes readings from MTR-99999, a meter not in the registry",
                    "About 0.4% of hourly readings are missing (gaps for rule R5)"],
        reference="reference/daily_usage_clean.csv: daily totals per meter before problems were injected"), files)

# ======================================================================
# W6. Riverton taxi trips: partitioned Parquet for Spark
# ======================================================================
def taxi():
    bid = "riverton-taxi-bigdata"; d = bundle_dir(bid); rng = np.random.default_rng(3606)
    os.makedirs(os.path.join(d, "trips"))
    zones = pd.DataFrame({"zone_id": range(1, 61), "zone_name": [f"{p} {s}" for p, s in zip(
        np.repeat(["Riverside", "Old Port", "Midtown", "Harbor", "Cedar", "Summit", "Lakeview", "Mill", "North End", "University", "Airport", "Garden"], 5),
        ["North", "South", "East", "West", "Central"] * 12)],
        "borough": np.repeat(["Riverside", "Old Port", "Midtown", "Harbor", "Cedar", "Summit", "Lakeview", "Mill", "North End", "University", "Airport", "Garden"], 5)})
    zones.to_csv(os.path.join(d, "zones.csv"), index=False)
    total = 0; samples = []
    pop = rng.dirichlet(np.ones(60) * .6)
    for m in range(1, 13):
        n = int(20000 * (1 + .15 * np.sin(2 * np.pi * (m - 4) / 12)))
        start = pd.Timestamp(f"2025-{m:02d}-01"); days = pd.Period(f"2025-{m:02d}").days_in_month
        hour = rng.choice(24, n, p=np.array([1, .6, .4, .3, .3, .5, 1.2, 2.2, 2.6, 2, 1.7, 1.8, 2, 1.9, 1.9, 2.1, 2.4, 2.8, 2.9, 2.6, 2.2, 1.9, 1.6, 1.3]) / 40.2)
        pu = start + pd.to_timedelta(rng.integers(0, days, n), unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
        dist = np.round(rng.gamma(1.8, 1.7, n), 2)
        dur = np.maximum(2, dist * rng.uniform(2.5, 5, n) + rng.normal(4, 2, n))
        fare = np.round(3 + 2.5 * dist + .4 * dur, 2)
        pay = rng.choice([1, 2, 3, 4], n, p=[.68, .27, .03, .02])
        tip = np.where(pay == 1, np.round(fare * rng.choice([0, .15, .18, .2, .25], n, p=[.15, .25, .3, .2, .1]), 2), 0)
        tolls = np.where(rng.random(n) < .05, 6.55, 0)
        df = pd.DataFrame({"trip_id": np.arange(total + 1, total + n + 1), "vendor": rng.choice(["RTX", "CityCab"], n, p=[.55, .45]),
                           "pickup_ts": pu, "dropoff_ts": pu + pd.to_timedelta(dur, unit="m"),
                           "pickup_zone": rng.choice(np.arange(1, 61), n, p=pop), "dropoff_zone": rng.choice(np.arange(1, 61), n, p=pop),
                           "passengers": rng.choice([1, 1, 1, 1, 2, 2, 3, 4, 5], n).astype("int8"), "distance_mi": dist,
                           "fare": fare, "tip": tip, "tolls": tolls, "payment_type": pay.astype("int8")})
        df["total"] = np.round(df.fare + df.tip + df.tolls + 1.0, 2)
        # data-quality problems typical of trip records
        k = rng.choice(n, int(n * .004), replace=False); df.loc[k, "passengers"] = 0
        k = rng.choice(n, int(n * .002), replace=False); df.loc[k, "fare"] = -df.loc[k, "fare"]
        k = rng.choice(n, int(n * .003), replace=False); df.loc[k, "distance_mi"] = 0.0
        k = rng.choice(n, int(n * .001), replace=False); df.loc[k, "dropoff_ts"] = df.loc[k, "pickup_ts"] - pd.Timedelta(minutes=5)
        k = rng.choice(n, int(n * .001), replace=False); df.loc[k, "distance_mi"] = df.loc[k, "distance_mi"] * 100
        part = os.path.join(d, "trips", f"month=2025-{m:02d}"); os.makedirs(part)
        df.to_parquet(os.path.join(part, "part-0.parquet"), index=False, compression="zstd")
        samples.append(df.sample(850, random_state=m))  # 10,200-row sample; total += n
    pd.concat(samples).sort_values("pickup_ts").to_csv(os.path.join(d, "sample_10k.csv"), index=False)
    open(os.path.join(d, "make_more_trips.py"), "w").write('''"""Generate a bigger version of the Riverton taxi data for Spark practice.
Usage:  python make_more_trips.py --rows 20000000 --out trips_big
Writes Parquet files partitioned by month (about 1 GB per 30 million rows)."""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=10_000_000); ap.add_argument("--out", default="trips_big")
ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
rng = np.random.default_rng(a.seed); per_month = a.rows // 12; chunk = 1_000_000; tid = 0
for m in range(1, 13):
    os.makedirs(f"{a.out}/month=2025-{m:02d}", exist_ok=True)
    for part, start in enumerate(range(0, per_month, chunk)):
        n = min(chunk, per_month - start)
        pu = pd.Timestamp(f"2025-{m:02d}-01") + pd.to_timedelta(rng.integers(0, 28 * 86400, n), unit="s")
        dist = np.round(rng.gamma(1.8, 1.7, n), 2); dur = np.maximum(2, dist * rng.uniform(2.5, 5, n) + 4)
        fare = np.round(3 + 2.5 * dist + .4 * dur, 2); pay = rng.choice([1, 2, 3, 4], n, p=[.68, .27, .03, .02])
        tip = np.where(pay == 1, np.round(fare * .18, 2), 0)
        df = pd.DataFrame({"trip_id": np.arange(tid, tid + n), "vendor": rng.choice(["RTX", "CityCab"], n), "pickup_ts": pu,
                           "dropoff_ts": pu + pd.to_timedelta(dur, unit="m"), "pickup_zone": rng.integers(1, 61, n),
                           "dropoff_zone": rng.integers(1, 61, n), "passengers": rng.choice([1, 1, 1, 2, 3, 4], n),
                           "distance_mi": dist, "fare": fare, "tip": tip, "tolls": 0.0, "payment_type": pay})
        df["total"] = df.fare + df.tip + 1.0; tid += n
        df.to_parquet(f"{a.out}/month=2025-{m:02d}/part-{part}.parquet", index=False)
    print("month", m, "done")
''')
    register(bid, dict(
        name="Riverton Taxi Trips (Big Data)", summary=f"{total:,} taxi trips in monthly Parquet partitions, built for Spark, with a script to scale to tens of millions.",
        description="A year of fictional taxi trips stored the way large datasets are stored in practice: compressed Parquet files partitioned by "
                    "month. Use it to learn Spark (or DuckDB / Polars) on a laptop, then run make_more_trips.py to generate 10, 20 or 50 million rows "
                    "and see why distributed processing matters. A 10,000-row CSV sample is included for quick exploration in pandas.",
        weeks=[5, 13], formats=["Parquet", "Partitioned data", "CSV"], level="Advanced",
        topics=["Spark", "Parquet", "Partitioning", "Outliers at scale", "Aggregation"],
        schema=[dict(name="trips/month=YYYY-MM/part-0.parquet", fields=[
            ("trip_id", "integer", "Trip ID"), ("vendor", "string", "RTX or CityCab"), ("pickup_ts / dropoff_ts", "timestamp", "Trip start and end"),
            ("pickup_zone / dropoff_zone", "integer", "Zone ID (see zones.csv)"), ("passengers", "integer", "Passenger count"),
            ("distance_mi", "float", "Trip distance in miles"), ("fare / tip / tolls / total", "float", "Amounts in USD (total adds a $1 surcharge)"),
            ("payment_type", "integer", "1 = card, 2 = cash, 3 = no charge, 4 = dispute")]),
            dict(name="zones.csv", fields=[("zone_id", "integer", "1-60"), ("zone_name", "string", "Zone"), ("borough", "string", "Borough")])],
        tasks=[(5, "In the 10k sample, find impossible trips: zero passengers, negative fares, zero distance with a fare, dropoff before pickup"),
               (13, "Load all partitions with Spark; count trips per month and compare read times with and without a month filter (partition pruning)"),
               (13, "Join zones.csv and compute revenue and average tip rate by borough and hour of day"),
               (13, "Generate 20 million rows with make_more_trips.py and repeat; compare pandas, Spark and DuckDB")],
        starter="""from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("riverton-taxi").getOrCreate()
# Download and unzip the bundle first, then point Spark at the folder
trips = spark.read.parquet("riverton-taxi-bigdata/trips")      # month is read from the folder names
zones = spark.read.csv("riverton-taxi-bigdata/zones.csv", header=True, inferSchema=True)

(trips.filter((F.col("fare") > 0) & (F.col("passengers") > 0))
      .groupBy("month").agg(F.count("*").alias("trips"), F.round(F.avg("total"), 2).alias("avg_total"))
      .orderBy("month").show())""",
        challenges=["0.4% of trips have zero passengers", "0.2% have negative fares", "0.3% have zero distance but a fare",
                    "0.1% end before they start", "0.1% have distances 100 times too large (unit or decimal error)",
                    "payment_type is a code: tips are only recorded for card payments, so 'no tip' on cash is not missing data"],
        reference="No clean reference: decide and document your filtering rules"),
        [("trips/month=2025-01/part-0.parquet", "January (12 monthly partitions in trips/)"), ("zones.csv", "60 taxi zones"),
         ("sample_10k.csv", "10,200-row sample for quick work in pandas"), ("make_more_trips.py", "Script to generate millions more rows")])

# ======================================================================
# W7. Riverton Climate Service: wide, messy-header exports
# ======================================================================
def climate():
    bid = "station-climate-wide"; d = bundle_dir(bid); rng = np.random.default_rng(3707)
    os.makedirs(os.path.join(d, "raw")); os.makedirs(os.path.join(d, "reference"))
    ns = 40
    st = pd.DataFrame({"station_id": [f"RCS{100+i}" for i in range(ns)],
                       "station_name": [f"{a}, {b}" for a, b in zip(rng.choice(["Riverton Airport", "Harbor Point", "Maple Ridge", "Cedar Falls", "Summit Peak", "Old Port Pier", "Lakeview", "Mill Creek", "North Fork", "Garden Hill"], ns), rng.choice(["NE", "SE", "MW", "SW", "NW"], ns))],
                       "lat": np.round(rng.uniform(30, 47, ns), 3), "lon": np.round(rng.uniform(-110, -72, ns), 3),
                       "elev_m": rng.integers(2, 2400, ns)})
    st["elev_ft"] = np.round(st.elev_m * 3.281).astype(int)
    years = range(1995, 2025); mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    rows, long = [], []
    for s in st.itertuples():
        base = 25 - .45 * (s.lat - 30) - .0065 * s.elev_m
        amp = 8 + .35 * (s.lat - 30)
        for y in years:
            trend = .028 * (y - 1995)
            vals = [base + trend - amp * np.cos(2 * np.pi * k / 12) + rng.normal(0, 1.1) for k in range(12)]
            prec = [max(0, rng.gamma(2, 35 + 15 * np.sin(2 * np.pi * (k - 3) / 12))) for k in range(12)]
            rows.append((s.station_id, s.station_name, y, vals, prec))
            for k in range(12):
                long.append(dict(station_id=s.station_id, year=y, month=k + 1, temp_c=round(vals[k], 1), precip_mm=round(prec[k], 1)))
    # temperatures: wide, tenths of °C, flags, sentinel, comment header lines
    lines = ["# Riverton Climate Service: monthly mean temperature export", "# Units: tenths of degrees Celsius (e.g. 123 = 12.3 °C)",
             "# Flags: E = estimated, M = missing month, -9999 = no data", "# Generated 2025-01-15. Do not edit.", ""]
    out = []
    for sid, name, y, vals, _ in rows:
        cells = []
        for v in vals:
            r = rng.random()
            if r < .01: cells.append("-9999")
            elif r < .02: cells.append("M")
            elif r < .06: cells.append(f"{int(round(v*10))}E")
            else: cells.append(str(int(round(v * 10))))
        annual = str(int(round(np.mean(vals) * 10)))
        out.append([sid, name, y] + cells + [annual])
    tdf = pd.DataFrame(out, columns=["station_id", "station_name", "year"] + mon + ["ANNUAL"])
    with open(os.path.join(d, "raw", "monthly_temperature_wide.csv"), "w") as f:
        f.write("\n".join(lines) + "\n"); tdf.to_csv(f, index=False)
    # precipitation: stations as rows, year-month as columns (the other orientation)
    pw = {}
    for sid, _, y, _, prec in rows:
        for k, p in enumerate(prec):
            pw.setdefault(sid, {})[f"{y}-{k+1:02d}"] = round(p, 1)
    pd.DataFrame(pw).T.rename_axis("station_id").reset_index().to_csv(os.path.join(d, "raw", "precipitation_by_month.csv"), index=False)
    # metadata: Excel with a two-row header
    with pd.ExcelWriter(os.path.join(d, "raw", "station_metadata.xlsx")) as xw:
        meta = st[["station_id", "station_name", "lat", "lon", "elev_m", "elev_ft"]].copy()
        meta.columns = pd.MultiIndex.from_tuples([("Station", "ID"), ("Station", "Name"), ("Location", "Latitude"), ("Location", "Longitude"),
                                                  ("Elevation", "m"), ("Elevation", "ft")])
        meta.to_excel(xw, sheet_name="Stations")
    pd.DataFrame(long).to_csv(os.path.join(d, "reference", "climate_long_clean.csv"), index=False)
    register(bid, dict(
        name="Station Climate Exports", summary="30 years of monthly temperature and rainfall for 40 stations, in three awkward wide layouts.",
        description="Exports from a fictional climate service, each in a different shape: temperatures with months as columns, comment lines at "
                    "the top, values in tenths of a degree and quality flags glued to the numbers; precipitation with one column per year-month "
                    "(360 columns); and station metadata in Excel with a two-row header. Reshape everything into one tidy long table.",
        weeks=[2, 5, 6], formats=["CSV (wide)", "Excel (multi-row header)"], level="Intermediate",
        topics=["read_csv options", "melt / pivot", "Multi-level headers", "Feature engineering", "Anomalies"],
        schema=[dict(name="raw/monthly_temperature_wide.csv", note="5 comment lines, then:", fields=[
                    ("station_id, station_name, year", "text", "Station names contain commas (quoted)"),
                    ("JAN … DEC", "text", "Tenths of °C; may end in 'E' (estimated), be 'M' or -9999"), ("ANNUAL", "integer", "Annual mean, tenths of °C")]),
                dict(name="raw/precipitation_by_month.csv", fields=[("station_id", "text", "Station"), ("1995-01 … 2024-12", "float", "Monthly precipitation in mm (360 columns)")]),
                dict(name="raw/station_metadata.xlsx", note="Sheet 'Stations' with a two-row header:", fields=[
                    ("Station / ID, Name", "text", "Station"), ("Location / Latitude, Longitude", "float", "Degrees"), ("Elevation / m, ft", "integer", "Elevation")])],
        tasks=[(2, "Read the temperature file with comment='#', and the Excel file with header=[0, 1]"),
               (5, "Separate the E flags, treat M and -9999 as missing, and convert tenths to degrees"),
               (6, "Melt both files to long format (station, year, month) and merge with the metadata"),
               (6, "Engineer features: seasonal means, anomaly versus the 1995-2004 baseline, 10-year trend per station")],
        starter="""import pandas as pd

BASE = "{BASE}raw/"
temps = pd.read_csv(BASE + "monthly_temperature_wide.csv", comment="#")
precip = pd.read_csv(BASE + "precipitation_by_month.csv")
meta = pd.read_excel(BASE + "station_metadata.xlsx", header=[0, 1], index_col=0)

long = temps.melt(id_vars=["station_id", "station_name", "year"],
                  value_vars=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
                  var_name="month", value_name="raw_value")
print(long.head())""",
        challenges=["The file starts with 5 comment lines (and a blank line) before the header",
                    "Values are tenths of °C stored as text because some end in 'E'",
                    "'M' and -9999 both mean missing", "Station names contain commas inside quotes",
                    "Precipitation is the opposite orientation: 360 year-month columns",
                    "The Excel sheet has a two-row header and an index column", "ANNUAL is a derived column that should not be melted with the months"],
        reference="reference/climate_long_clean.csv: one row per station-month with temperature (°C) and precipitation (mm)"),
        [("raw/monthly_temperature_wide.csv", "Temperatures, months as columns, with comments and flags"),
         ("raw/precipitation_by_month.csv", "Precipitation, 360 year-month columns"), ("raw/station_metadata.xlsx", "Station metadata, two-row header"),
         ("reference/climate_long_clean.csv", "Reference: tidy long table")])

# ======================================================================
# W8. Web server logs: regex parsing, sessionization, JSON lines
# ======================================================================
def logs():
    bid = "webserver-logs"; d = bundle_dir(bid); rng = np.random.default_rng(3808)
    os.makedirs(os.path.join(d, "logs")); os.makedirs(os.path.join(d, "reference"))
    pagesl = ["/", "/courses", "/courses/data-wrangling", "/courses/machine-learning", "/library", "/library/search", "/login",
              "/dashboard", "/assignments", "/assignments/submit", "/api/v1/grades", "/api/v1/calendar", "/static/app.js", "/static/style.css",
              "/images/logo.png", "/about", "/contact", "/favicon.ico"]
    pw = np.array([10, 6, 4, 4, 5, 3, 6, 8, 5, 3, 4, 3, 9, 9, 7, 2, 1, 5], float); pw /= pw.sum()
    agents = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
              'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
              'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148',
              'Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0']
    bots = ['Googlebot/2.1 (+http://www.google.com/bot.html)', 'Mozilla/5.0 (compatible; bingbot/2.0)', 'python-requests/2.32', 'curl/8.6.0']
    users = [f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(1500)]
    users += [f"2001:db8:{rng.integers(0, 65535):x}::{rng.integers(1, 9999):x}" for _ in range(150)]
    ua_of = {u: agents[rng.integers(len(agents))] for u in users}
    bot_ips = [f"66.249.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(12)]
    events = []; parsed = []
    for day in pd.date_range("2025-09-01", "2025-09-07"):
        lines = []
        nsess = int(rng.normal(2600, 150) * (0.6 if day.dayofweek >= 5 else 1))
        for _ in range(nsess):
            ip = users[rng.integers(len(users))]; bot = False
            if rng.random() < .06: ip = bot_ips[rng.integers(len(bot_ips))]; bot = True
            t = day + pd.Timedelta(seconds=float(rng.uniform(6 * 3600, 24 * 3600 - 1)))
            for _ in range(int(rng.integers(1, 12))):
                path = str(rng.choice(pagesl, p=pw)); method = "POST" if path in ("/login", "/assignments/submit") else "GET"
                status = int(rng.choice([200, 200, 200, 200, 304, 301, 404, 500, 403], p=[.55, .15, .08, .04, .1, .03, .03, .01, .01]))
                size = 0 if status in (304,) else int(rng.lognormal(8, 1.2))
                rid = f"{rng.integers(16**8):08x}"; ms = int(rng.lognormal(4.3, .8))
                ua = bots[rng.integers(len(bots))] if bot else ua_of[ip]
                ts = t.strftime("%d/%b/%Y:%H:%M:%S -0400")
                ref = "-" if rng.random() < .4 else f"https://learn.example.edu{rng.choice(pagesl)}"
                lines.append(f'{ip} - - [{ts}] "{method} {path} HTTP/1.1" {status} {size} "{ref}" "{ua}" rt={ms}ms rid={rid}')
                if len(parsed) < 5000:
                    parsed.append(dict(ip=ip, timestamp=t.strftime("%Y-%m-%d %H:%M:%S"), method=method, path=path, status=status, bytes=size,
                                       referrer=None if ref == "-" else ref, user_agent=ua, response_ms=ms, request_id=rid, is_bot=bot))
                if path.startswith("/api") or path in ("/login", "/assignments/submit"):
                    events.append({"request_id": rid, "ts": t.isoformat() + "-04:00", "event": {"/login": "auth.login", "/assignments/submit": "assignment.submit"}.get(path, "api.call"),
                                   "context": {"user": {"id": f"u{abs(hash(ip)) % 90000 + 10000}", "role": str(rng.choice(["student", "student", "student", "instructor"]))},
                                               "client": {"ip": ip, "app_version": str(rng.choice(["3.2.0", "3.2.1", "3.3.0"]))}},
                                   "outcome": {"ok": status < 400, "status": status, "latency_ms": ms}})
                t += pd.Timedelta(seconds=float(rng.exponential(45)))
        # malformed and foreign lines
        for _ in range(int(len(lines) * .002)):
            lines.insert(int(rng.integers(len(lines))), rng.choice([
                f'[{day.strftime("%a %b %d")} 03:12:{rng.integers(10, 59)} 2025] [error] [client {users[0]}] File does not exist: /var/www/wp-login.php',
                '', '"GET /broken HTTP/1.1" 400', f'{users[1]} - - [{day.strftime("%d/%b/%Y")}:99:99:99 -0400] "GET / HTTP/1.1" 200 512 "-" "-"']))
        with gzip.open(os.path.join(d, "logs", f"access-{day.strftime('%Y-%m-%d')}.log.gz"), "wt") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "app_events.jsonl"), "w") as f:
        for e in events: f.write(json.dumps(e) + "\n")
    pd.DataFrame(parsed).to_csv(os.path.join(d, "reference", "parsed_first_5000.csv"), index=False)
    register(bid, dict(
        name="Learning Platform Web Logs", summary="A week of web-server access logs plus application events in JSON Lines, ready for regex parsing and sessionization.",
        description="Seven days of compressed access logs (about 96,000 requests) from a fictional online learning platform, in the Apache combined "
                    "format with two extra fields (response time and request ID). A separate JSON Lines file records application events with nested "
                    "context, linked to the logs by request ID. Parse, clean, sessionize, filter bots and join the two sources.",
        weeks=[3, 5, 14], formats=["Log files (gzip)", "JSON Lines", "Regex"], level="Advanced",
        topics=["Regular expressions", "Sessionization", "Bot filtering", "Chunked reading", "JSON Lines"],
        schema=[dict(name="logs/access-YYYY-MM-DD.log.gz", note="One request per line:", fields=[
                    ("ip", "text", "IPv4 or IPv6"), ("[timestamp]", "text", "dd/Mon/yyyy:HH:MM:SS -0400"), ('"METHOD path HTTP/1.1"', "text", "Request line"),
                    ("status, bytes", "integer", "HTTP status and response size"), ('"referrer" "user agent"', "text", "Quoted strings"),
                    ("rt=…ms rid=…", "text", "Response time and request ID")]),
                dict(name="app_events.jsonl", note="One JSON object per line:", fields=[
                    ("request_id", "text", "Links to rid= in the logs"), ("ts", "text", "ISO 8601"), ("event", "text", "auth.login, assignment.submit or api.call"),
                    ("context.user / context.client", "object", "{id, role} and {ip, app_version}"), ("outcome", "object", "{ok, status, latency_ms}")])],
        tasks=[(3, "Write one regular expression that parses every valid line into columns; count the lines it rejects"),
               (5, "Flag bots by user agent and request rate; find the slowest pages (p95 response time)"),
               (14, "Sessionize: a new session starts after 30 minutes of inactivity per IP; report pages per session"),
               (14, "Read app_events.jsonl with pd.read_json(lines=True), flatten the nested context and join to the logs on request ID")],
        starter="""import gzip, re, urllib.request, pandas as pd

LINE = re.compile(r'(?P<ip>\\S+) \\S+ \\S+ \\[(?P<ts>[^\\]]+)\\] "(?P<method>\\S+) (?P<path>\\S+) \\S+" '
                  r'(?P<status>\\d{3}) (?P<bytes>\\d+) "(?P<ref>[^"]*)" "(?P<ua>[^"]*)" rt=(?P<ms>\\d+)ms rid=(?P<rid>\\w+)')

urllib.request.urlretrieve("{BASE}logs/access-2025-09-01.log.gz", "day1.log.gz")
rows, bad = [], 0
with gzip.open("day1.log.gz", "rt") as f:
    for line in f:
        m = LINE.match(line)
        rows.append(m.groupdict()) if m else (bad := bad + 1)
logs = pd.DataFrame(rows)
print(len(logs), "parsed,", bad, "rejected")""",
        challenges=["About 0.2% of lines are not access-log lines: error-log entries, empty lines and truncated requests",
                    "Some lines have impossible times (99:99:99)", "About 10% of clients use IPv6 addresses",
                    "About 6% of sessions come from crawlers and scripts (Googlebot, bingbot, python-requests, curl)",
                    "Timestamps carry a -0400 offset that must be kept or converted consistently",
                    "Status 304 responses have 0 bytes legitimately"],
        reference="reference/parsed_first_5000.csv: the first 5,000 valid requests of day 1, parsed"),
        [("logs/access-2025-09-01.log.gz", "Day 1 access log, gzip (7 files in logs/)"), ("app_events.jsonl", "Application events, one JSON object per line"),
         ("reference/parsed_first_5000.csv", "Reference: first 5,000 requests parsed")])

if __name__ == "__main__":
    shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
    for f in [events_api, books_web, retail, hospital_db, meter_pipeline, taxi, climate, logs]:
        f()
    json.dump(CATALOG, open(os.path.join(HERE, "wrangling_catalog.json"), "w"), indent=1, default=str)
    tot = sum(os.path.getsize(os.path.join(b, f)) for b, _, fs in os.walk(ROOT) for f in fs)
    print(len(CATALOG), "bundles; folder total", round(tot / 1024 / 1024, 1), "MB (including zips)")
