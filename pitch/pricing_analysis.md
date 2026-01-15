# Pricing Analysis
## Player Tracking & Simulation Platform

---

## Market Context

### What Competitors Charge

| Platform | Target Market | Annual Price | Per-Match Equivalent |
|----------|---------------|--------------|---------------------|
| SkillCorner | Pro clubs | $80K-200K | $2,000-5,000 |
| Second Spectrum | Pro leagues | $100K-300K | $3,000-8,000 |
| Catapult (GPS) | All levels | $50K-100K | Hardware + subscription |
| Stats Perform | Pro clubs | $60K-150K | $1,500-4,000 |
| Hudl Sportscode | College | $15K-30K | Manual tagging required |

**Key Insight:** No affordable automated tracking solution exists for college soccer. Hudl requires manual work; everything else is priced for pro budgets.

---

## Our Cost Structure

### Variable Costs (Per Match)

| Item | Cost | Notes |
|------|------|-------|
| GPU compute | $15-40 | ~90 min processing on cloud GPU |
| Storage | $1-2 | Video + output data |
| **Total per match** | **$20-45** | |

### Fixed Costs (Monthly)

| Item | Cost | Notes |
|------|------|-------|
| Cloud infrastructure | $100-200 | Base servers, databases |
| Model hosting | $50-100 | Inference endpoints |
| **Total monthly fixed** | **$150-300** | |

### Annual Operating Cost

| Scenario | Matches | Annual Cost |
|----------|---------|-------------|
| Light (home games only) | 10-12 | $2,000-3,000 |
| Standard (all matches) | 20-25 | $4,000-6,000 |
| Full (matches + training) | 50+ | $8,000-15,000 |

---

## Pricing Models

### Option A: Internal Cost Recovery

*If you just need to justify budget internally*

| Tier | Matches/Year | Budget Request |
|------|--------------|----------------|
| Pilot | 5 | $500 |
| Basic | 15 | $3,000 |
| Standard | 25 | $6,000 |
| Premium | 50+ | $12,000 |

**Positioning:** "We can deliver SkillCorner-equivalent analytics for 5% of their cost."

---

### Option B: Service Pricing (If Selling to Other Programs)

*If you want to offer this to other schools/teams*

**Per-Match Pricing:**
| Tier | Price/Match | Includes |
|------|-------------|----------|
| Basic | $150-250 | Tracking data + basic report |
| Standard | $300-500 | + tactical analysis + video overlay |
| Premium | $600-1,000 | + simulation + custom dashboards |

**Season Packages:**
| Package | Price | Matches | Per-Match |
|---------|-------|---------|-----------|
| Starter | $2,500 | 10 | $250 |
| Conference | $5,000 | 20 | $250 |
| Full Season | $8,000 | 35 | $229 |
| All-Access | $15,000 | Unlimited | — |

**Gross Margin:** 70-85% at these prices

---

### Option C: Freemium / Land-and-Expand

*Best for building adoption*

| Tier | Price | Access |
|------|-------|--------|
| Free Trial | $0 | 2 matches, basic tracking only |
| Essentials | $3,000/yr | 15 matches, standard reports |
| Pro | $8,000/yr | Unlimited matches, full features |
| Enterprise | Custom | Multi-team, API access, custom models |

---

## Recommended Approach for Marshall

### Phase 1: Prove Value (Free/Internal)

- Run pilot at cost (~$500)
- No need to discuss pricing externally
- Focus on demonstrating capabilities

### Phase 2: Operational Budget

- Request $6,000-10,000 annual budget
- Position as 95% savings vs. commercial options
- Include in standard video/analytics budget line

### Phase 3: Optional Revenue (Later)

If Marshall wants to monetize:
- Offer to Sun Belt rivals at $5,000-8,000/season
- Partner with regional clubs
- License to other Conference USA programs

---

## Pricing Positioning Statements

**For Budget Approval:**
> "Commercial player tracking costs $80,000+ annually. We can build equivalent capability in-house for under $10,000/year—a 90% cost reduction while giving Marshall a competitive edge no other Sun Belt program has."

**For Selling to Others:**
> "Professional-grade player tracking for collegiate budgets. Full season analysis for less than the cost of one recruiting trip."

**Value Anchors:**
- SkillCorner = $100K+ → We're $8K (92% less)
- Cost per insight = cents vs. dollars
- ROI: One prevented injury or successful recruit pays for years of service

---

## Bottom Line Recommendations

| Situation | Recommended Price |
|-----------|-------------------|
| Marshall internal pilot | $500 (cost only) |
| Marshall full season | $8,000-12,000/year |
| Selling to other D1 programs | $6,000-10,000/year |
| Selling to D2/D3/clubs | $3,000-5,000/year |
| Per-match one-offs | $300-500/match |

**My suggestion:** Start with a free pilot to prove value, then request $8K-10K annual budget. That's defensible, delivers massive value, and leaves room to expand if you want to offer it to other programs later.
