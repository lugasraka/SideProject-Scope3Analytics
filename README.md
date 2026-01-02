# 🏢 CarbonTrace AI: Scope 3 Embodied Carbon Estimator & Optimizer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An AI-powered web application for estimating and optimizing Scope 3 upstream embodied carbon emissions in building portfolios. Built with machine learning models to help facility managers make data-driven sustainability decisions.

## 🚀 [**Try the Live Demo**](https://huggingface.co/spaces/lugasraka/Scope3BuildingAnalytics)

---

## 📋 Problem Statement

**The Challenge:** Scope 3 emissions—particularly embodied carbon from building materials—are often the largest and most difficult portion of a company's carbon footprint to measure and reduce. Many facility managers lack visibility into upstream emissions across their building portfolios, making it difficult to:

- Identify high-emission assets and carbon hotspots
- Fill data gaps for buildings without embodied carbon measurements
- Evaluate the carbon impact of material choices in retrofit scenarios
- Make informed decisions about sustainable construction practices

**The Gap:** Traditional carbon accounting methods are time-consuming, expensive, and require specialized expertise. Most organizations struggle to estimate embodied carbon at scale or simulate the impact of material substitutions.

---

## 🎯 Project Overview

**CarbonTrace AI** is an interactive web application that empowers sustainability professionals to:

1. **Visualize Portfolio Emissions**: See total embodied carbon across building portfolios with interactive dashboards and geographic maps
2. **Estimate Missing Data**: Use AI to predict Scope 3 emissions for buildings without measured data
3. **Optimize Retrofit Decisions**: Compare the carbon impact of different structural materials before construction
4. **Benchmark Models**: Understand which machine learning models perform best and what factors drive emissions

The tool focuses on **upstream Scope 3 emissions** from the supply chain, including materials extraction, manufacturing, and transportation.

---

## 🎓 Objective

**Primary Goal**: Democratize embodied carbon analysis by providing an accessible, AI-driven tool that enables non-technical users to:

- Gain instant insights into building portfolio emissions
- Make evidence-based decisions on material selection
- Identify opportunities for carbon reduction
- Support LEED/BREEAM certification goals

**Target Users**: Facility managers, sustainability consultants, architects, real estate developers, and corporate ESG teams.

---

## 🤖 Machine Learning Models

The application compares three ensemble learning algorithms to predict embodied carbon emissions:

### 1. **Random Forest**
- **Type**: Bagging ensemble of decision trees
- **Strengths**: Robust, handles non-linear relationships, resistant to overfitting
- **Use Case**: Best for general-purpose predictions with good balance of speed and accuracy

### 2. **Gradient Boosting**
- **Type**: Sequential ensemble that corrects errors iteratively
- **Strengths**: Often highest accuracy, captures complex patterns
- **Trade-offs**: Slower training, more sensitive to hyperparameters

### 3. **Extra Trees (Extremely Randomized Trees)**
- **Type**: Randomized ensemble with increased variance
- **Strengths**: Fastest training, comparable accuracy to Random Forest
- **Use Case**: Ideal when speed is critical or for large datasets

### Input Features:
- **GFA_sqm**: Gross Floor Area (square meters)
- **Num_Floors**: Number of building floors
- **Building_Type**: Office, Residential, Education, Healthcare
- **Structure_Type**: Concrete, Steel, Timber, Mixed
- **Year_Built**: Construction year
- **Geographic Data**: Latitude, Longitude (for visualization only)

### Target Variable:
- **Actual_Scope3_tCO2e**: Total upstream embodied carbon in tonnes CO₂ equivalent

---

## 📊 Key Results

Based on the current dataset (10,000+ buildings):

### Model Performance:
| Model | MAE (tCO2e) | R² Score | RMSE | Training Time |
|-------|-------------|----------|------|---------------|
| **Extra Trees** | ~550 | 0.95+ | ~700 | Fastest |
| **Random Forest** | ~560 | 0.95+ | ~710 | Fast |
| **Gradient Boosting** | ~565 | 0.95+ | ~720 | Moderate |

### Feature Importance (Top Drivers):
1. **GFA_sqm (70-75%)**: Building size is the dominant predictor of embodied carbon
2. **Structure_Type_Timber (10-15%)**: Material choice significantly impacts emissions
3. **Structure_Type_Mixed (4-5%)**: Mixed materials contribute moderate emissions
4. **Num_Floors (3-5%)**: Vertical complexity affects material requirements
5. **Building_Type variations (2-3%)**: Different uses have varying carbon intensities

### Key Insights:
- **Building size** accounts for ~70% of embodied carbon variability
- **Material selection** (timber vs. steel vs. concrete) can change emissions by 30-40%
- **Extra Trees** model provides the best speed-accuracy trade-off for this application
- Models achieve **95%+ R²** on test data, indicating strong predictive power

---

## 🔬 Methodology

### 1. Data Collection & Generation
- Synthetic dataset of 10,000+ buildings with realistic distributions
- Features engineered from industry standards (GFA, structure types, building uses)
- Geographic coordinates for portfolio visualization

### 2. Data Preprocessing
- One-hot encoding for categorical variables (Building_Type, Structure_Type)
- Feature scaling handled inherently by tree-based models
- 80/20 train-test split with random_state=42 for reproducibility

### 3. Model Training & Evaluation
- Three ensemble models trained and compared
- Evaluation metrics: MAE, RMSE, R² Score, Training Time
- Feature importance extraction for interpretability

### 4. Application Development
- **Framework**: Gradio for interactive web UI
- **Visualization**: Plotly for dynamic charts and maps
- **Deployment**: Standalone Python app, easily shareable

### 5. Model Validation
- Actual vs. Predicted scatter plots for each model
- Cross-comparison of performance metrics
- Feature importance ranking with explanations

---

## 💼 Business Implications

### For Facility Managers:
- **Cost Savings**: Identify opportunities to reduce embodied carbon through material substitution
- **Risk Mitigation**: Understand carbon liabilities across portfolio before regulatory mandates
- **Certification Support**: Quantify carbon reductions for LEED, BREEAM, or WELL certifications

### For Sustainability Teams:
- **Data Gap Filling**: Estimate emissions for buildings without lifecycle assessments
- **Portfolio Prioritization**: Focus resources on highest-impact buildings
- **Scenario Planning**: Model different retrofit strategies before committing resources

### For Real Estate Developers:
- **Pre-Construction Decisions**: Compare material options during design phase
- **ESG Reporting**: Improve Scope 3 disclosure accuracy and completeness
- **Competitive Advantage**: Demonstrate carbon leadership to tenants and investors

### ROI Examples:
- **Material Switching**: Replacing steel with timber can reduce emissions by 30-40%
- **Avoided Costs**: Potential carbon pricing at $50-100/tonne makes low-carbon materials economically attractive
- **Faster Decision-Making**: AI estimates in seconds vs. weeks for traditional LCA

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/lugasraka/SideProject-Scope3Analytics.git
cd SideProject-Scope3Analytics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Or use Gradio CLI:

```bash
gradio app.py
```

The application will launch in your default web browser at `http://localhost:7860`

---

## 📱 Application Features

### Tab 1: Portfolio Overview
- Total emissions metrics across all buildings
- Carbon intensity by building type (bar chart)
- Carbon intensity by structure type (violin plot)
- **Interactive Map**: Geographic visualization with hover details for each building

### Tab 2: AI Estimator (Gap Filling)
- Input building parameters (GFA, floors, type, year, structure)
- Select prediction model
- Get instant embodied carbon estimate

### Tab 3: Retrofit Optimizer
- Compare baseline vs. proposed materials
- Visualize emissions reduction potential
- Calculate carbon savings and ROI
- Get business case recommendations

### Tab 4: Model Comparison
- Compare all three models side-by-side
- View performance metrics (MAE, R², RMSE)
- See actual vs. predicted scatter plots
- Explore feature importance rankings with explanations

---

## 📁 Project Structure

```
SideProject-Scope3Analytics/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── df_train-new-location.csv       # Training dataset with locations
├── data-generator.ipynb            # Notebook for generating synthetic data
├── CarbonTrace-Scope3.ipynb        # Exploratory analysis notebook
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🔮 Future Enhancements

- [ ] Integration with real building data APIs (Energy Star, LEED)
- [ ] Add operational carbon (Scope 1 & 2) estimates
- [ ] Time-series forecasting for portfolio emissions trends
- [ ] Export reports to PDF/Excel for stakeholder sharing
- [ ] Multi-user authentication and team collaboration
- [ ] Integration with carbon offset marketplaces
- [ ] Mobile-responsive design improvements

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Raka Adrianto**  
PM Sustainability  
[LinkedIn](https://www.linkedin.com/in/lugasraka/)

---

## 📚 References

- [GHG Protocol Scope 3 Standard](https://ghgprotocol.org/scope-3-frequently-asked-questions-0)
- [LEED v4.1 Building Design and Construction](https://www.usgbc.org/leed/v41)
- [Carbon Leadership Forum](https://carbonleadershipforum.org/)
- [Embodied Carbon in Construction Calculator (EC3)](https://www.buildingtransparency.org/)

---

## ⚠️ Disclaimer

This tool is designed for educational and preliminary analysis purposes. For official carbon accounting and reporting, please consult with certified sustainability professionals and use validated lifecycle assessment (LCA) methodologies.

---

**Live Since**: December 2025  
**Version**: 1.0.0
