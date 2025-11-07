# CS 4100 Final Project Roadmap
## RL vs Human Financial Strategies

### **CURRENT STATUS: MVP COMPLETE ✅**
- Environment, agents, training, evaluation all working
- Need to add statistical rigor + analysis for paper quality

---

## **WEEK 1: CORE EXPERIMENTS & DATA COLLECTION**
*Deadline: Tuesday, November 12*

### **Person 1: Statistical Foundation (15-20 hours)**
**Tasks:**
- Modify evaluation script to run 5 seeds per agent
- Implement statistical tests (t-tests, Cohen's d)
- Add error bars and confidence intervals to plots
- Create statistical significance table

**Deliverables:**
- `run_statistical_analysis.py` 
- Updated plots with error bars
- Statistical results table (p-values, effect sizes)

**Time Estimate:** 3-4 days

### **Person 2: Behavioral Analysis (15-20 hours)**
**Tasks:**
- Track agent actions by market regime (bull/bear/neutral)
- Analyze allocation patterns (stocks vs bonds vs emergency fund)
- Create action heatmaps showing strategy differences
- Implement market timing analysis

**Deliverables:**
- `behavioral_analysis.py`
- Action comparison visualizations
- Market timing analysis plots

**Time Estimate:** 3-4 days

### **Person 3: Ablation Study + SAC Agent (15-20 hours)**
**Tasks:**
- Implement SAC agent (copy PPO structure, change algorithm)
- Create stable market environment (no volatility)
- Run ablation experiment: RL performance with/without market volatility
- Add financial metrics (Sharpe ratio, max drawdown)

**Deliverables:**
- `sac_agent.py`
- `stable_market_env.py` 
- Ablation study results
- Financial risk metrics

**Time Estimate:** 4-5 days

---

## **WEEK 2: ANALYSIS & INSIGHTS**
*Deadline: Tuesday, November 19*

### **All Team: Data Analysis (20-25 hours total)**
**Tasks:**
- Run full experimental suite (11 agents × 5 seeds = 55 runs)
- Analyze why RL outperforms baselines
- Identify learned strategies vs rule-based strategies
- Calculate all financial performance metrics

**Deliverables:**
- Complete experimental results
- Insights document: "Why RL Works Better"
- Performance comparison tables

**Time Estimate:** 5-6 days

---

## **WEEK 3: PAPER & PRESENTATION**
*Deadline: December 1 (Presentation), December 10 (Final Report)*

### **Person 1: Paper Writing (20-25 hours)**
**Tasks:**
- Write Abstract, Introduction, Related Work
- Methods section (environment, agents, evaluation)
- Format in LaTeX, create bibliography

### **Person 2: Results & Analysis (20-25 hours)**
**Tasks:**
- Results section with all plots and tables
- Discussion section analyzing findings
- Conclusion and future work

### **Person 3: Code & Presentation (20-25 hours)**
**Tasks:**
- Clean up GitHub repository
- Write comprehensive README
- Create presentation slides
- Record demo video

**Deliverables:**
- 6-page technical paper
- 15-minute presentation
- Polished GitHub repository

---

## **DETAILED EXPERIMENTS**

### **EXPERIMENT 1: Statistical Significance**
```python
# What: Run each agent 5 times, calculate statistics
# Output: "DQN significantly outperforms Markowitz (p < 0.001, d = 1.2)"
# Time: 2 days coding + 1 day running experiments
```

### **EXPERIMENT 2: Behavioral Analysis**
```python
# What: Track agent actions in different market conditions
# Output: "RL reduces stock allocation 15% during bear markets, baselines don't adapt"
# Time: 3 days coding + analysis
```

### **EXPERIMENT 3: Ablation Study**
```python
# What: Remove market volatility, see if RL advantage disappears
# Output: "RL advantage drops from 59% to 12% without market volatility"
# Time: 2 days environment modification + 1 day experiments
```

---

## **STATISTICAL TESTS NEEDED**

- **T-tests**: Is difference significant? (p < 0.05)
- **Cohen's d**: How big is the effect? (d > 0.8 = large)
- **Confidence intervals**: Error bars on all plots
- **Multiple comparisons**: Bonferroni correction for 11 agents

---

## **BEHAVIORAL INSIGHTS NEEDED**

- **Market timing**: Do RL agents reduce risk during crashes?
- **Debt management**: Do they prioritize high-interest debt?
- **Emergency fund**: Do they build larger safety nets?
- **Allocation patterns**: How do strategies differ by age/market?

---

## **FINANCIAL METRICS NEEDED**

- **Sharpe ratio**: Risk-adjusted returns
- **Max drawdown**: Worst loss period
- **Volatility**: Standard deviation of returns
- **Success rate**: % achieving retirement goals

---

## **TOTAL TIME ESTIMATE**

- **Week 1**: 45-60 hours (15-20 per person)
- **Week 2**: 20-25 hours (collaborative analysis)
- **Week 3**: 60-75 hours (20-25 per person)
- **TOTAL**: 125-160 hours across 3 people over 3 weeks

---

## **COMMANDS TO RUN**

### **Current MVP Results:**
```bash
# Train RL agents
python train_agents.py both 1000

# Evaluate all agents
cd evaluation
python evaluate_agents.py
```

### **Week 1 Additions:**
```bash
# Statistical analysis
python run_statistical_analysis.py

# Behavioral analysis
python behavioral_analysis.py

# Ablation study
python ablation_study.py
```

---

## **SUCCESS CRITERIA**

**Paper Quality Equivalent to Equiloan:**
- ✅ Working RL environment with multiple agents
- ✅ Baseline comparisons showing RL superiority
- 🔄 Statistical significance testing
- 🔄 Behavioral analysis of learned strategies
- 🔄 Ablation study proving why RL works
- 🔄 6-page technical report
- 🔄 15-minute presentation

**Research Contribution:**
*"First study showing RL can learn superior long-term financial strategies compared to traditional rule-based approaches, with 59% improvement in 30-year net worth outcomes."*
