# AI
To use `pgvector` for your real estate project on Railway, you are essentially turning your database into a "Search Engine" that understands the *meaning* of property descriptions, not just the keywords.

Here is the detailed, step-by-step guide to setting it up, seeding it, and using it in your 2026 AI project.

---

### 1. Setup: The Foundation (Railway & SQL)

Since you are on Railway, you don't need to manually install software. You just need to enable the capability.

#### **A. Deploy the Database**

In your Railway project, click **New** > **Database** > **Add PostgreSQL**.

* **Pro Tip:** Look for the **"Postgres with pgVector"** template in the Railway Marketplace. It comes pre-optimized.

#### **B. Enable the Extension**

Connect to your database (using the Railway "Query" tab or a tool like DBeaver) and run:

```sql
-- This activates the vector math engine
CREATE EXTENSION IF NOT EXISTS vector;

```

#### **C. Create the Schema**

We will create a table that holds both **Standard Data** (Price, BHK) and **Vector Data** (The "Vibe").

> **Note on Dimensions:** For **Gemini Embedding 2**, the standard size is **3072**, but Google recommends **768** as the "sweet spot" for speed and cost.

```sql
CREATE TABLE properties (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    price INTEGER,
    location TEXT,
    bhk INTEGER,
    description TEXT,
    -- The 'vector(768)' column stores the AI representation
    embedding vector(768) 
);

-- Speed up searches using an HNSW index (The 2026 standard for fast AI search)
CREATE INDEX ON properties USING hnsw (embedding vector_cosine_ops);

```

---

### 2. Seeding: Feeding the AI

"Seeding" is the process of taking your existing property list and generating embeddings for each one so the database can "understand" them.

#### **The Workflow:**

1. **Extract:** Get the text description of the property.
2. **Embed:** Send that text to the **Gemini API**.
3. **Store:** Save the resulting array of numbers into the `embedding` column.

**Node.js Seeding Script Example:**

```javascript
import { GoogleGenAI } from "@google/genai";
import pg from 'pg';

const genAI = new GoogleGenAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-embedding-2-preview" });

async function seedProperty(property) {
    // 1. Generate the Embedding
    const result = await model.embedContent(property.description);
    const vector = result.embedding.values; // This is an array of 768 numbers

    // 2. Insert into PostgreSQL
    const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
    await pool.query(
        'INSERT INTO properties (title, price, description, embedding) VALUES ($1, $2, $3, $4)',
        [property.title, property.price, property.description, JSON.stringify(vector)]
    );
}

```

---

### 3. Usage: The "Vibe Search"

When a user asks your bot for a home, you don't search for "3BHK Gurgaon." You search for the **meaning** of their request.

#### **The "Hybrid" Query**

This is how you find a home that matches a **Budget** (SQL) AND a **Vibe** (Vector).

```javascript
async function searchProperties(userQuery, maxPrice) {
    // 1. Turn the user's question into a vector
    const result = await model.embedContent(userQuery);
    const queryVector = JSON.stringify(result.embedding.values);

    // 2. Run the Hybrid Search
    const sql = `
        SELECT title, price, location, 
               1 - (embedding <=> $1) AS similarity_score
        FROM properties
        WHERE price <= $2
        ORDER BY embedding <=> $1
        LIMIT 5;
    `;
    
    // <=> is the Cosine Distance operator
    const { rows } = await pool.query(sql, [queryVector, maxPrice]);
    return rows;
}

```

---

### Why this is a "Leads Machine" for Delhi-NCR:

* **The Problem:** A buyer says, *"I want a house like the one in 'Zindagi Na Milegi Dobara'."* A standard search fails.
* **The Solution:** Your AI-powered `pgvector` knows that movie's vibe is "Mediterranean, luxury, open spaces." It will find properties in **Gurgaon** or **Noida** that mention "Spanish architecture" or "infinity pools" even if the user didn't use those words.

### Summary Checklist for your Project:

1. **Railway:** Deploy Postgres + Enable `vector` extension.
2. **API:** Get your Gemini API Key for the `gemini-embedding-2-preview` model.
3. **App:** Use the `<=>` operator in your SQL to find similar homes.

**Would you like me to create a "Scraper" script that can take a property PDF brochure and automatically turn it into these database entries for you?**
