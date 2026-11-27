# Theoretical Analysis: Assignment 7 — Car Tracking (HMM) & Logic

**Author:** Muzira Joshua Kwizera (Student No. 2500703484)  
**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)

---

## 1. Hidden Markov Models (HMM) & Particle Filtering

### 1.1 Forward-Backward Algorithm Derivation
Let $H_t \in \mathcal{S}$ be the hidden car position at step $t$ and $E_t$ the noisy distance observation:
- **Filtering / Forward Pass:**
  $$P(H_t \mid e_{1:t}) \propto P(e_t \mid H_t) \sum_{h_{t-1} \in \mathcal{S}} P(H_t \mid h_{t-1}) P(h_{t-1} \mid e_{1:t-1})$$
- **Time Complexity:** Computing one forward step over $|\mathcal{S}|$ states takes $O(|\mathcal{S}|^2)$ time (or $O(|\mathcal{S}| \cdot k)$ when each state has at most $k$ transitions). Over $T$ steps, total exact inference is $O(T |\mathcal{S}|^2)$.

### 1.2 Particle Depletion & Importance Resampling
When $|\mathcal{S}|$ is large (e.g., high-resolution grids or multi-car tracking), exact inference becomes intractable. **Particle Filtering** maintains $N$ unweighted particles:
1. **Propose (Elapse Time):** Sample $h_t^{(i)} \sim P(H_t \mid h_{t-1}^{(i)})$.
2. **Weight (Observe):** $w^{(i)} = P(e_t \mid h_t^{(i)})$.
3. **Resample:** Draw $N$ new particles proportional to $w^{(i)}$ (systematic resampling). Without resampling, all weight concentrates on a single particle (**particle degeneracy / depletion**); resampling focuses compute on high-posterior regions.

---

## 2. Propositional & First-Order Logic

### 2.1 Conjunctive Normal Form (CNF) Transformation
Any propositional formula is converted to an equivalent conjunction of disjunctions of literals via four steps:
1. **Eliminate $\leftrightarrow$ and $\to$:** Replace $A \to B$ with $\neg A \lor B$, and $A \leftrightarrow B$ with $(\neg A \lor B) \land (\neg B \lor A)$.
2. **Move $\neg$ inward (De Morgan's Laws):** $\neg(A \land B) \equiv \neg A \lor \neg B$, $\neg(A \lor B) \equiv \neg A \land \neg B$, and eliminate double negations $\neg\neg A \equiv A$.
3. **Distribute $\lor$ over $\land$:** $A \lor (B \land C) \equiv (A \lor B) \land (A \lor C)$.

### 2.2 Resolution Proofs: Soundness & Completeness
Given two clauses $(A \lor p)$ and $(B \lor \neg p)$, **Resolution** derives $(A \lor B)$:
- **Soundness:** If $(A \lor p)$ and $(B \lor \neg p)$ are both true under any truth assignment, then whether $p$ is true or false, $(A \lor B)$ must be true.
- **Refutation Completeness:** To prove $\text{KB} \models \alpha$, we convert $\text{KB} \land \neg \alpha$ to CNF and repeatedly apply resolution. By Robinson's Resolution Theorem, $\text{KB} \models \alpha$ if and only if the empty clause $\square$ is derivable.
