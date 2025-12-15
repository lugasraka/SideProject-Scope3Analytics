
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Loading Data and Training Engine with Machine Learning Models ---
# Objective: Predict 'Actual_Scope3_tCO2e' based on features
# Features: 'GFA_sqm', 'Building_Type', 'Num_Floors', 'Year_Built', 'Structure_Type'
# ------------------------------------------------------------

def load_data():
    """Load training data from CSV"""
    return pd.read_csv('https://github.com/lugasraka/SideProject-Scope3Analytics/blob/fe5f47720fe8059e77b96f0c9dee2d6415c68570/df_train-new-location.csv?raw=true')

def train_model(df, model_name='Random Forest'):
    """Trains the selected Estimator"""
    # Preprocessing
    df_ml = pd.get_dummies(df, columns=['Building_Type', 'Structure_Type'], drop_first=False)
    # Drop non-numeric location columns and target variables
    cols_to_drop = ['Actual_Scope3_tCO2e', 'Intensity_kgCO2e_m2', 'City', 'Country']
    X = df_ml.drop(cols_to_drop, axis=1)
    y = df_ml['Actual_Scope3_tCO2e']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    elif model_name == 'Extra Trees':
        model = ExtraTreesRegressor(n_estimators=50, random_state=42)
    else:
        raise ValueError("Invalid model name")

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, X.columns, mae, r2, y_test, y_pred, rmse

# Load Data
df_train = load_data()


# --- Gradio App ---
# Objective: Create an interactive web app for Portfolio Overview, AI Estimator, Retrofit Optimizer, and Model Comparison
# ------------------------------------------------------------

# Portfolio Overview Tab
# Goal: Show key metrics and visualizations of the portfolio
def portfolio_overview():
    total_emissions = df_train['Actual_Scope3_tCO2e'].sum()
    avg_intensity = df_train['Intensity_kgCO2e_m2'].mean()
    assets_tracked = len(df_train)

    hotspots = df_train.groupby('Building_Type', observed=True)['Actual_Scope3_tCO2e'].sum().reset_index()
    fig1 = px.bar(hotspots, x='Building_Type', y='Actual_Scope3_tCO2e', color='Building_Type', labels={'Actual_Scope3_tCO2e':'Total tCO2e'}, title='Total Upstream Carbon by Building Type')
    
    fig2 = px.violin(
        df_train,
        x='Structure_Type',
        y='Intensity_kgCO2e_m2',
        color='Structure_Type',
        box=True,
        title='Carbon Intensity by Structure Type (Violin + Points)'
    )
    fig2.update_traces(meanline_visible=True)

    # Create geographic map visualization
    try:
        # Filter out rows with missing latitude/longitude and emissions data
        df_map = df_train.dropna(subset=['Latitude', 'Longitude', 'Actual_Scope3_tCO2e', 'Intensity_kgCO2e_m2']).copy()
        
        # Ensure numeric types
        df_map['Latitude'] = pd.to_numeric(df_map['Latitude'], errors='coerce')
        df_map['Longitude'] = pd.to_numeric(df_map['Longitude'], errors='coerce')
        df_map['Actual_Scope3_tCO2e'] = pd.to_numeric(df_map['Actual_Scope3_tCO2e'], errors='coerce')
        df_map['Intensity_kgCO2e_m2'] = pd.to_numeric(df_map['Intensity_kgCO2e_m2'], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        df_map = df_map.dropna(subset=['Latitude', 'Longitude', 'Actual_Scope3_tCO2e', 'Intensity_kgCO2e_m2'])
        
        if len(df_map) > 0:
            # Create custom hover text to avoid formatting issues
            df_map['Hover_Text'] = df_map.apply(
                lambda row: f"<b>{row['City']}, {row['Country']}</b><br>" +
                           f"Type: {row['Building_Type']}<br>" +
                           f"GFA: {row['GFA_sqm']:,.0f} m²<br>" +
                           f"Floors: {int(row['Num_Floors'])}<br>" +
                           f"Year Built: {int(row['Year_Built'])}<br>" +
                           f"Structure: {row['Structure_Type']}<br>" +
                           f"Scope 3: {row['Actual_Scope3_tCO2e']:,.0f} tCO2e<br>" +
                           f"Intensity: {row['Intensity_kgCO2e_m2']:.0f} kgCO2e/m²",
                axis=1
            )
            
            fig_map = px.scatter_map(
                df_map,
                lat='Latitude',
                lon='Longitude',
                hover_name='Hover_Text',
                size='Actual_Scope3_tCO2e',
                color='Intensity_kgCO2e_m2',
                color_continuous_scale='RdYlGn_r',
                map_style='open-street-map',
                title='Building Locations & Embodied Carbon Intensity',
                zoom=2,
                center={'lat': 50, 'lon': 10},
                labels={'Intensity_kgCO2e_m2': 'Intensity (kgCO2e/m²)'}
            )
            fig_map.update_layout(
                height=600,
                hovermode='closest'
            )
        else:
            fig_map = go.Figure()
            fig_map.add_annotation(text="No location data available")
    except Exception as e:
        fig_map = go.Figure()
        fig_map.add_annotation(text=f"Error creating map: {str(e)}")

    return f"{total_emissions:,.0f} tCO2e", f"{avg_intensity:.0f} kgCO2e/m²", f"{assets_tracked}", fig1, fig2, fig_map

# AI Estimator Tab
# Goal: Estimate embodied carbon for buildings without data
def ai_estimator(model_name, gfa, floors, btype, year, struct):
    # Train model on demand
    model, model_cols, _, _, _, _, _ = train_model(df_train, model_name)

    # Prepare Input Vector
    input_data = pd.DataFrame([{
        'GFA_sqm': float(gfa),
        'Building_Type': btype,
        'Num_Floors': float(floors),
        'Year_Built': float(year),
        'Structure_Type': struct
    }])
    
    # Encoding (One-Hot) to match training data
    input_encoded = pd.get_dummies(input_data, columns=['Building_Type', 'Structure_Type'])
    # Reindex guarantees correct columns and order; missing cols filled with 0
    input_encoded = input_encoded.reindex(columns=model_cols, fill_value=0)
    
    # Predict
    prediction = model.predict(input_encoded)[0]
    
    return f"{prediction:,.0f} tCO2e", f"Based on historical data for {struct} {btype} buildings of this size."

# Retrofit Optimizer Tab
# Goal: Compare embodied carbon of different structural materials for retrofit scenarios
def retrofit_optimizer(model_name, base_gfa, base_floors, base_type, mat_A, mat_B):
    # Train model on demand
    model, model_cols, _, _, _, _, _ = train_model(df_train, model_name)

    def get_pred(mat):
        row = pd.DataFrame([{
            'GFA_sqm': base_gfa, 'Building_Type': base_type, 
            'Num_Floors': base_floors, 'Year_Built': 2024, 'Structure_Type': mat
        }])
        enc = pd.get_dummies(row, columns=['Building_Type', 'Structure_Type'], drop_first=False)
        # Reindex guarantees correct columns and order; missing cols filled with 0
        enc = enc.reindex(columns=model_cols, fill_value=0)
        return model.predict(enc)[0]

    pred_A = get_pred(mat_A)
    pred_B = get_pred(mat_B)
    
    savings = pred_A - pred_B
    pct_savings = (savings / pred_A) * 100 if pred_A > 0 else 0
    
    chart_data = pd.DataFrame({
        'Scenario': [f"Baseline ({mat_A})", f"Retrofit ({mat_B})"],
        'Emissions': [pred_A, pred_B]
    })
    fig = px.bar(chart_data, x='Scenario', y='Emissions', color='Scenario', color_discrete_map={f"Baseline ({mat_A})":"#FF4B4B", f"Retrofit ({mat_B})":"#00CC96"}, title='Retrofit Comparison')
    fig.update_layout(yaxis_title='Total Upstream Carbon (tCO2e)')
    
    business_case = ""
    if pred_A > 0:
        if pct_savings > 0:
            business_case = f"💡 **Business Case:** Switching to {mat_B} reduces embodied carbon by {pct_savings:.1f}%. This can contribute to LEED/BREEAM certification points."
        else:
            business_case = "⚠️ The proposed material increases or does not reduce emissions in this specific scenario."
    else:
        business_case = "Model returned zero baseline emissions — check inputs or retrain model."
            
    return f"{pred_A:,.0f} tCO2e", f"{pred_B:,.0f} tCO2e", f"{savings:,.0f} tCO2e ({pct_savings:.1f}%)", fig, business_case

# Model Comparison Tab
# Goal: Compare model performance and training time and then recommend the best model (balance of accuracy and speed)
def compare_models():
    """Compares model performance, runtime, and highlights top feature drivers."""
    models = ['Random Forest', 'Gradient Boosting', 'Extra Trees']
    results = []
    plots = []
    
    for model_name in models:
        start_time = time.time()
        _, cols, mae, r2, y_test, y_pred, rmse = train_model(df_train, model_name)
        end_time = time.time()
        training_time = end_time - start_time
        results.append({'Model': model_name, 'Training Time (s)': round(training_time, 3), 'MAE (tCO2e)': round(mae, 3), 'R2 Score': round(r2, 3), 'RMSE (tCO2e)': round(rmse, 3)})
        
        # Create scatter plot
        fig_scatter = px.scatter(x=y_test, y=y_pred, 
                                 title=f'{model_name}: Actual vs. Predicted (R2 = {r2:.3f}, RMSE = {rmse:.0f})',
                                 labels={'x': 'Actual Emissions (tCO2e)', 'y': 'Predicted Emissions (tCO2e)'})
        # Add a line for perfect correlation
        fig_scatter.add_shape(type='line',
                              x0=y_test.min(), y0=y_test.min(),
                              x1=y_test.max(), y1=y_test.max(),
                              line=dict(color='Red', dash='dash'))
        plots.append(fig_scatter)

    df_results = pd.DataFrame(results)
    
    # Create figure with secondary y-axis for bar chart
    fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for bar chart
    fig_bar.add_trace(
        go.Bar(x=df_results['Model'], y=df_results['MAE (tCO2e)'], name='MAE (tCO2e)'),
        secondary_y=False,
    )

    fig_bar.add_trace(
        go.Scatter(x=df_results['Model'], y=df_results['Training Time (s)'], name='Training Time (s)'),
        secondary_y=True,
    )

    # Add figure title for bar chart
    fig_bar.update_layout(
        title_text="Model Performance vs. Training Time"
    )

    # Set x-axis title for bar chart
    fig_bar.update_xaxes(title_text="Model")

    # Set y-axes titles for bar chart
    fig_bar.update_yaxes(title_text="<b>MAE (tCO2e)</b> - Lower is Better", secondary_y=False)
    fig_bar.update_yaxes(title_text="<b>Training Time (s)</b> - Lower is Better", secondary_y=True)

    # Identify best model (lowest MAE)
    best_row = df_results.sort_values('MAE (tCO2e)').iloc[0]
    best_model_name = best_row['Model']

    # Train best model to extract feature importance
    best_model, model_cols, _, _, _, _, _ = train_model(df_train, best_model_name)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': model_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        total_imp = feat_df['Importance'].sum()
        feat_df['Percent'] = (feat_df['Importance'] / total_imp * 100).round(1)
        # Drop the 'Importance' column to show only percentage
        feat_df = feat_df[['Feature', 'Percent']]
    else:
        feat_df = pd.DataFrame({'Feature': model_cols, 'Percent': np.nan})

    # Build interpretation text for top drivers with explanations
    feature_explanations = {
        'GFA_sqm': 'Larger buildings require more materials, directly increasing embodied carbon.',
        'Num_Floors': 'More floors demand greater structural material volume and complexity.',
        'Year_Built': 'Older buildings often used less efficient materials; newer ones may have lower-carbon alternatives.',
        'Structure_Type_': 'Material choice (steel vs. concrete vs. timber) significantly impacts total embodied carbon.',
        'Building_Type_': 'Different building types have varying material intensities based on design requirements.'
    }
    
    def get_explanation(feature_name):
        # Direct match first
        if feature_name in feature_explanations:
            return feature_explanations[feature_name]
        # Check if feature starts with any key (for one-hot encoded features)
        for key, explanation in feature_explanations.items():
            if feature_name.startswith(key):
                return explanation
        return 'Key driver of embodied carbon emissions.'
    
    top_feats = feat_df.head(5)
    driver_lines = [f"**{idx+1}. {row['Feature']}** — {row['Percent']:.1f}%: \n   {get_explanation(row['Feature'])}" 
                    for idx, (_, row) in enumerate(top_feats.iterrows())]
    driver_md = "**Top feature drivers (by importance), including their interpretations**\n\n" + "\n\n".join(driver_lines)

    recommendation = f"""### Model Recommendation

- **Random Forest:** Robust baseline; good balance of accuracy and speed.
- **Gradient Boosting:** Often highest accuracy but slower; sensitive to tuning.
- **Extra Trees:** Fastest among the three; can match RF accuracy on many datasets.

**Current best model with the lowest MAE (Mean Absolute Error): {best_model_name}.**"""
    
    return df_results, fig_bar, recommendation, plots[0], plots[1], plots[2], feat_df, driver_md


with gr.Blocks(theme=gr.themes.Default(), title="CarbonTrace Upstream MVP") as demo:
    gr.Markdown("# 🏢 CarbonTrace AI: Scope 3 Impact Estimator & Optimizer for Buildings Portfolio")
    gr.Markdown("""
        **Product Vision:** Empower Facility Managers to visualize 'hidden' Scope 3 emissions and simulate retrofit scenarios using AI/ML, enabling data-driven sustainability decisions.
            Scope 3 emissions are the embodied carbon emissions from the full lifecycle of buildings, including materials extraction, manufacturing, transportation, construction, maintenance, and end-of-life disposal.
            For this demo, we focused on the 'Upstream' portion of Scope 3 emissions from the supply chain/embodied emissions of materials.
            While the tool only covers a subset of Scope 3 emissions, it provides critical insights into the often overlooked embodied carbon in buildings, which can constitute a significant portion of a company's total carbon footprint.
            For more details of Scope 3 emissions, check out the FAQ page of the [GHG Protocol](https://ghgprotocol.org/scope-3-frequently-asked-questions-0).
        
        **Author:** [Raka Adrianto](https://www.linkedin.com/in/lugasraka/), PM Sustainability, Sustainability

        **Live since:** December 2025
        """)

    with gr.Tabs():
        with gr.TabItem("1. Portfolio Overview"):
            with gr.Row():
                total_emissions_out = gr.Textbox(label="Total Upstream Carbon")
                avg_intensity_out = gr.Textbox(label="Avg Intensity")
                assets_tracked_out = gr.Textbox(label="Assets Tracked")
            with gr.Row():
                plot1_out = gr.Plot()
                plot2_out = gr.Plot()
            with gr.Row():
                map_out = gr.Plot(label="Geographic Distribution of Buildings")
            
            load_button = gr.Button("Load Portfolio Data")
            load_button.click(
                fn=portfolio_overview, 
                outputs=[total_emissions_out, avg_intensity_out, assets_tracked_out, plot1_out, plot2_out, map_out]
            )

        with gr.TabItem("2. AI Estimator (Gap Filling)"):
            gr.Markdown("Enter basic details for a building with **NO** embodied carbon data. The AI will estimate it.")
            with gr.Row():
                in_model = gr.Dropdown(label="Select Model", choices=['Random Forest', 'Gradient Boosting', 'Extra Trees'], value='Random Forest')
            with gr.Row():
                with gr.Column():
                    in_gfa = gr.Number(label="GFA (sqm)", value=15000)
                    in_floors = gr.Number(label="Number of Floors", value=5)
                with gr.Column():
                    in_type = gr.Dropdown(label="Building Type", choices=['Office', 'Residential', 'Education', 'Healthcare'], value='Office')
                    in_year = gr.Number(label="Year Built", value=2010)
                with gr.Column():
                    in_struct = gr.Dropdown(label="Structure Material", choices=['Concrete', 'Steel', 'Timber', 'Mixed'], value='Concrete')
            
            run_button = gr.Button("Run AI Estimate")
            
            with gr.Row():
                ai_prediction_out = gr.Textbox(label="Predicted Scope 3 Upstream")
                ai_caption_out = gr.Textbox(label="Notes")

            run_button.click(
                fn=ai_estimator,
                inputs=[in_model, in_gfa, in_floors, in_type, in_year, in_struct],
                outputs=[ai_prediction_out, ai_caption_out]
            )

        with gr.TabItem("3. Retrofit Optimizer"):
            gr.Markdown("Quantify the carbon ROI of switching structural materials for a planned project.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Scenario Inputs")
                    in_model_retrofit = gr.Dropdown(label="Select Model", choices=['Random Forest', 'Gradient Boosting', 'Extra Trees'], value='Random Forest')
                    base_gfa = gr.Slider(label="Building Size (m²)", minimum=2000, maximum=50000, value=10000, step=100)
                    base_floors = gr.Slider(label="Floors", minimum=1, maximum=20, value=8, step=1)
                    base_type = gr.Dropdown(label="Building Use", choices=['Office', 'Residential', 'Education', 'Healthcare'], value='Office')
                    
                    gr.Markdown("---")
                    gr.Markdown("**Comparison**")
                    mat_A = gr.Dropdown(label="Baseline Material", choices=['Steel', 'Concrete', 'Timber', 'Mixed'], value='Steel')
                    mat_B = gr.Dropdown(label="Proposed Material", choices=['Timber', 'Mixed', 'Concrete', 'Steel'], value='Timber')

                with gr.Column():
                    gr.Markdown("### Simulation Results")
                    with gr.Row():
                        metric1_out = gr.Textbox(label="Baseline")
                        metric2_out = gr.Textbox(label="Proposed")
                        metric3_out = gr.Textbox(label="Net Avoided Carbon")
                    
                    retrofit_plot_out = gr.Plot()
                    business_case_out = gr.Markdown()

            inputs = [in_model_retrofit, base_gfa, base_floors, base_type, mat_A, mat_B]
            outputs = [metric1_out, metric2_out, metric3_out, retrofit_plot_out, business_case_out]

            base_gfa.release(retrofit_optimizer, inputs=inputs, outputs=outputs)
            base_floors.release(retrofit_optimizer, inputs=inputs, outputs=outputs)
            base_type.change(retrofit_optimizer, inputs=inputs, outputs=outputs)
            mat_A.change(retrofit_optimizer, inputs=inputs, outputs=outputs)
            mat_B.change(retrofit_optimizer, inputs=inputs, outputs=outputs)
            in_model_retrofit.change(retrofit_optimizer, inputs=inputs, outputs=outputs)

        with gr.TabItem("4. Model Comparison"):
            gr.Markdown("Compare the performance (Mean Absolute Error, R2 Score, RMSE) and training cost (time) of the available models.")
            run_comparison_btn = gr.Button("Run Model Comparison")
            
            with gr.Row():
                comparison_df = gr.DataFrame(label="Model Comparison Results")
            with gr.Row():
                comparison_plot = gr.Plot()
            with gr.Row():
                recommendation_out = gr.Markdown()
            with gr.Row():
                scatter_rf = gr.Plot()
                scatter_gb = gr.Plot()
                scatter_et = gr.Plot()
            with gr.Row():
                feature_table = gr.DataFrame(label="Top Feature Importance (Best Model)")
            with gr.Row():
                feature_notes = gr.Markdown()
                
            run_comparison_btn.click(
                fn=compare_models,
                outputs=[comparison_df, comparison_plot, recommendation_out, scatter_rf, scatter_gb, scatter_et, feature_table, feature_notes]
            )


if __name__ == "__main__":
    demo.launch()
