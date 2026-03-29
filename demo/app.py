"""Streamlit demo for fraud detection system.

This module provides an interactive Streamlit application for fraud detection
with real-time predictions and explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fraud_detection import (
    TransactionDataGenerator, 
    FraudDetectionPipeline,
    FraudDetectionEvaluator
)
from fraud_detection.viz import FraudExplainer, FraudDetectionPlots


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add disclaimer
    st.warning("""
    **DISCLAIMER**: This system is for research and educational purposes only. 
    It should not be used for automated decision-making without human review.
    """)
    
    # Title
    st.title("🔍 Fraud Detection System")
    st.markdown("Advanced fraud detection for business operations")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    
    # Sidebar options
    st.sidebar.subheader("Data Generation")
    n_transactions = st.sidebar.slider("Number of transactions", 1000, 10000, 5000)
    fraud_rate = st.sidebar.slider("Fraud rate", 0.01, 0.3, 0.1)
    
    if st.sidebar.button("Generate Data"):
        with st.spinner("Generating transaction data..."):
            generator = TransactionDataGenerator(random_state=42, fraud_rate=fraud_rate)
            st.session_state.data = generator.generate_transactions(n_transactions)
            st.session_state.pipeline = None
            st.session_state.evaluator = None
        st.success(f"Generated {n_transactions} transactions with {fraud_rate:.1%} fraud rate")
    
    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🤖 Model Training", 
            "🔍 Predictions", 
            "📈 Evaluation", 
            "💡 Explanations"
        ])
        
        with tab1:
            show_data_overview(data)
        
        with tab2:
            show_model_training(data)
        
        with tab3:
            show_predictions(data)
        
        with tab4:
            show_evaluation(data)
        
        with tab5:
            show_explanations(data)
    
    else:
        st.info("👈 Please generate data first using the sidebar options.")


def show_data_overview(data: pd.DataFrame):
    """Show data overview and statistics."""
    st.header("📊 Data Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(data):,}")
    
    with col2:
        fraud_count = data['is_fraud'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    
    with col3:
        fraud_rate = data['is_fraud'].mean()
        st.metric("Fraud Rate", f"{fraud_rate:.1%}")
    
    with col4:
        avg_amount = data['amount'].mean()
        st.metric("Average Amount", f"${avg_amount:.2f}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10))
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud distribution
        fraud_counts = data['is_fraud'].value_counts()
        fig = px.pie(values=fraud_counts.values, names=['Normal', 'Fraud'], 
                     title="Transaction Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount distribution
        fig = px.histogram(data, x='amount', color='is_fraud', 
                          title="Transaction Amount Distribution",
                          labels={'is_fraud': 'Fraud Status'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud by category
        category_fraud = data.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=True)
        fig = px.bar(x=category_fraud.values, y=category_fraud.index, 
                     orientation='h', title="Fraud Rate by Merchant Category")
        fig.update_layout(xaxis_title="Fraud Rate", yaxis_title="Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud by hour
        if 'hour' in data.columns:
            hour_fraud = data.groupby('hour')['is_fraud'].mean()
            fig = px.line(x=hour_fraud.index, y=hour_fraud.values, 
                         title="Fraud Rate by Hour of Day")
            fig.update_layout(xaxis_title="Hour", yaxis_title="Fraud Rate")
            st.plotly_chart(fig, use_container_width=True)


def show_model_training(data: pd.DataFrame):
    """Show model training interface."""
    st.header("🤖 Model Training")
    
    if st.button("Train Models"):
        with st.spinner("Training fraud detection models..."):
            # Prepare data
            from fraud_detection.data.processor import DataProcessor
            
            processor = DataProcessor()
            X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
            y = data['is_fraud']
            
            # Split data
            X_train, X_test, y_train, y_test = processor.split_data(
                processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
            )
            
            # Train pipeline
            pipeline = FraudDetectionPipeline(random_state=42)
            pipeline.fit(X_train, y_train, X_test, y_test)
            
            # Store in session state
            st.session_state.pipeline = pipeline
            st.session_state.evaluator = FraudDetectionEvaluator(pipeline)
            
        st.success("Models trained successfully!")
    
    if st.session_state.pipeline is not None:
        st.subheader("Model Performance")
        
        # Get model performance
        performance = st.session_state.pipeline.get_model_performance()
        
        if not performance.empty:
            # Display performance metrics
            st.dataframe(performance, use_container_width=True)
            
            # Performance visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('AUC', 'Precision', 'Recall', 'F1 Score'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            metrics = ['auc', 'precision', 'recall', 'f1']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for metric, (row, col) in zip(metrics, positions):
                fig.add_trace(
                    go.Bar(x=performance['model'], y=performance[metric], name=metric.upper()),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = st.session_state.pipeline.get_feature_importance()
        
        if not feature_importance.empty:
            # Top features across all models
            top_features = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=True).tail(15)
            
            fig = px.bar(x=top_features.values, y=top_features.index, 
                         orientation='h', title="Top 15 Features by Average Importance")
            fig.update_layout(xaxis_title="Average Importance", yaxis_title="Feature")
            st.plotly_chart(fig, use_container_width=True)


def show_predictions(data: pd.DataFrame):
    """Show prediction interface."""
    st.header("🔍 Fraud Detection Predictions")
    
    if st.session_state.pipeline is None:
        st.warning("Please train the models first.")
        return
    
    # Prediction options
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_mode = st.selectbox(
            "Prediction Mode",
            ["Single Transaction", "Batch Predictions", "Real-time Simulation"]
        )
    
    with col2:
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5)
    
    if prediction_mode == "Single Transaction":
        show_single_prediction(threshold)
    elif prediction_mode == "Batch Predictions":
        show_batch_predictions(data, threshold)
    else:
        show_realtime_simulation(threshold)


def show_single_prediction(threshold: float):
    """Show single transaction prediction interface."""
    st.subheader("Single Transaction Prediction")
    
    # Input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.01, value=100.0)
            frequency = st.number_input("Daily Transaction Frequency", min_value=1, value=5)
            is_foreign = st.selectbox("Foreign Transaction", [0, 1])
            is_high_risk_country = st.selectbox("High Risk Country", [0, 1])
        
        with col2:
            is_weekend = st.selectbox("Weekend Transaction", [0, 1])
            is_night = st.selectbox("Night Transaction", [0, 1])
            merchant_category = st.selectbox("Merchant Category", 
                                           ['grocery', 'gas_station', 'restaurant', 'retail', 'online'])
            device_type = st.selectbox("Device Type", ['mobile', 'desktop', 'tablet'])
        
        submitted = st.form_submit_button("Predict Fraud")
    
    if submitted:
        # Create transaction data
        transaction_data = pd.DataFrame([{
            'amount': amount,
            'frequency': frequency,
            'is_foreign': is_foreign,
            'is_high_risk_country': is_high_risk_country,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'merchant_category': merchant_category,
            'device_type': device_type,
            'channel': 'online',
            'is_high_risk_merchant': 1 if merchant_category in ['online', 'travel'] else 0
        }])
        
        # Make prediction
        fraud_prob = st.session_state.pipeline.predict_proba(transaction_data)[0]
        prediction = 1 if fraud_prob >= threshold else 0
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fraud Probability", f"{fraud_prob:.3f}")
        
        with col2:
            st.metric("Prediction", "🚨 FRAUD" if prediction else "✅ NORMAL")
        
        with col3:
            risk_level = "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.3 else "LOW"
            st.metric("Risk Level", risk_level)
        
        # Risk factors
        st.subheader("Risk Factors")
        risk_factors = []
        if is_foreign:
            risk_factors.append("Foreign transaction")
        if is_high_risk_country:
            risk_factors.append("High-risk country")
        if is_weekend:
            risk_factors.append("Weekend transaction")
        if is_night:
            risk_factors.append("Night transaction")
        if amount > 1000:
            risk_factors.append("High amount")
        if merchant_category in ['online', 'travel']:
            risk_factors.append("High-risk merchant")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"⚠️ {factor}")
        else:
            st.write("✅ No significant risk factors identified")


def show_batch_predictions(data: pd.DataFrame, threshold: float):
    """Show batch prediction interface."""
    st.subheader("Batch Predictions")
    
    # Sample data for prediction
    sample_size = st.slider("Sample Size", 100, 1000, 500)
    sample_data = data.sample(n=sample_size, random_state=42)
    
    if st.button("Run Batch Predictions"):
        with st.spinner("Running batch predictions..."):
            # Prepare data
            X = sample_data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
            y_true = sample_data['is_fraud']
            
            # Make predictions
            y_pred = st.session_state.pipeline.predict(X, threshold=threshold)
            y_pred_proba = st.session_state.pipeline.predict_proba(X)
            
            # Create results DataFrame
            results = sample_data.copy()
            results['predicted_fraud'] = y_pred
            results['fraud_probability'] = y_pred_proba
            
            # Display results
            st.subheader("Prediction Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = (y_pred == y_true).mean()
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with col2:
                precision = (y_true & y_pred).sum() / y_pred.sum() if y_pred.sum() > 0 else 0
                st.metric("Precision", f"{precision:.3f}")
            
            with col3:
                recall = (y_true & y_pred).sum() / y_true.sum() if y_true.sum() > 0 else 0
                st.metric("Recall", f"{recall:.3f}")
            
            with col4:
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                st.metric("F1 Score", f"{f1:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            confusion_data = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
            st.dataframe(confusion_data)
            
            # Results table
            st.subheader("Detailed Results")
            st.dataframe(results[['amount', 'merchant_category', 'is_fraud', 'predicted_fraud', 'fraud_probability']].head(20))


def show_realtime_simulation(threshold: float):
    """Show real-time simulation interface."""
    st.subheader("Real-time Simulation")
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        simulation_duration = st.slider("Simulation Duration (seconds)", 10, 60, 30)
        transaction_rate = st.slider("Transactions per second", 1, 10, 3)
    
    with col2:
        fraud_rate = st.slider("Simulated Fraud Rate", 0.01, 0.2, 0.05)
        alert_threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.7)
    
    if st.button("Start Simulation"):
        # Create placeholder for real-time updates
        placeholder = st.empty()
        
        # Simulation data
        generator = TransactionDataGenerator(random_state=42, fraud_rate=fraud_rate)
        
        # Run simulation
        import time
        start_time = time.time()
        transactions = []
        alerts = []
        
        while time.time() - start_time < simulation_duration:
            # Generate transaction
            transaction = generator.generate_transactions(1).iloc[0]
            
            # Make prediction
            X = pd.DataFrame([transaction.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'])])
            fraud_prob = st.session_state.pipeline.predict_proba(X)[0]
            
            # Check for alerts
            if fraud_prob >= alert_threshold:
                alerts.append({
                    'time': time.time() - start_time,
                    'amount': transaction['amount'],
                    'fraud_prob': fraud_prob,
                    'actual_fraud': transaction['is_fraud']
                })
            
            transactions.append({
                'time': time.time() - start_time,
                'amount': transaction['amount'],
                'fraud_prob': fraud_prob,
                'actual_fraud': transaction['is_fraud']
            })
            
            # Update display
            with placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Transactions", len(transactions))
                
                with col2:
                    st.metric("Alerts Generated", len(alerts))
                
                with col3:
                    if len(alerts) > 0:
                        alert_rate = len(alerts) / len(transactions)
                        st.metric("Alert Rate", f"{alert_rate:.1%}")
                
                # Real-time plot
                if len(transactions) > 10:
                    df_sim = pd.DataFrame(transactions)
                    fig = px.scatter(df_sim, x='time', y='fraud_prob', 
                                   color='actual_fraud', 
                                   title="Real-time Fraud Probability")
                    fig.add_hline(y=alert_threshold, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(1 / transaction_rate)
        
        st.success("Simulation completed!")
        
        # Final results
        if alerts:
            st.subheader("Generated Alerts")
            alerts_df = pd.DataFrame(alerts)
            st.dataframe(alerts_df)


def show_evaluation(data: pd.DataFrame):
    """Show model evaluation interface."""
    st.header("📈 Model Evaluation")
    
    if st.session_state.evaluator is None:
        st.warning("Please train the models first.")
        return
    
    if st.button("Run Comprehensive Evaluation"):
        with st.spinner("Running evaluation..."):
            # Prepare test data
            from fraud_detection.data.processor import DataProcessor
            
            processor = DataProcessor()
            X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
            y = data['is_fraud']
            
            # Split data
            X_train, X_test, y_train, y_test = processor.split_data(
                processor.prepare_features(pd.concat([X, y], axis=1), fit=True)
            )
            
            # Run evaluation
            results = st.session_state.evaluator.evaluate(X_test, y_test, X_test)
            
            # Display results
            st.subheader("Evaluation Results")
            
            # ML Metrics
            st.subheader("Machine Learning Metrics")
            ml_metrics = results['ml_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AUC", f"{ml_metrics['roc_auc']:.3f}")
            
            with col2:
                st.metric("Precision", f"{ml_metrics['precision']:.3f}")
            
            with col3:
                st.metric("Recall", f"{ml_metrics['recall']:.3f}")
            
            with col4:
                st.metric("F1 Score", f"{ml_metrics['f1_score']:.3f}")
            
            # Business Metrics
            st.subheader("Business Metrics")
            business_metrics = results['business_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Fraud Detection Rate", f"{business_metrics['fraud_detection_rate']:.1%}")
            
            with col2:
                st.metric("Alert Rate", f"{business_metrics['alert_rate']:.1%}")
            
            with col3:
                st.metric("Precision at K", f"{business_metrics['precision_at_k']:.1%}")
            
            with col4:
                st.metric("ROI", f"{business_metrics['roi']:.1%}")
            
            # Threshold Analysis
            st.subheader("Threshold Analysis")
            threshold_data = results['threshold_analysis']
            
            # Find optimal threshold (maximize F1 score)
            optimal_idx = threshold_data['f1_score'].idxmax()
            optimal_threshold = threshold_data.loc[optimal_idx, 'threshold']
            
            st.write(f"**Optimal Threshold**: {optimal_threshold:.2f} (F1 Score: {threshold_data.loc[optimal_idx, 'f1_score']:.3f})")
            
            # Threshold plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision vs Threshold', 'Recall vs Threshold', 
                              'F1 Score vs Threshold', 'Specificity vs Threshold')
            )
            
            fig.add_trace(go.Scatter(x=threshold_data['threshold'], y=threshold_data['precision'], 
                                   mode='lines+markers', name='Precision'), row=1, col=1)
            fig.add_trace(go.Scatter(x=threshold_data['threshold'], y=threshold_data['recall'], 
                                   mode='lines+markers', name='Recall'), row=1, col=2)
            fig.add_trace(go.Scatter(x=threshold_data['threshold'], y=threshold_data['f1_score'], 
                                   mode='lines+markers', name='F1 Score'), row=2, col=1)
            fig.add_trace(go.Scatter(x=threshold_data['threshold'], y=threshold_data['specificity'], 
                                   mode='lines+markers', name='Specificity'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False, title_text="Threshold Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Model Leaderboard
            st.subheader("Model Leaderboard")
            leaderboard = st.session_state.evaluator.get_leaderboard()
            st.dataframe(leaderboard, use_container_width=True)


def show_explanations(data: pd.DataFrame):
    """Show model explanations interface."""
    st.header("💡 Model Explanations")
    
    if st.session_state.pipeline is None:
        st.warning("Please train the models first.")
        return
    
    # Explanation options
    col1, col2 = st.columns(2)
    
    with col1:
        explanation_type = st.selectbox(
            "Explanation Type",
            ["Global Feature Importance", "Individual Prediction", "Sample Analysis"]
        )
    
    with col2:
        sample_size = st.slider("Sample Size for Analysis", 100, 1000, 500)
    
    if st.button("Generate Explanations"):
        with st.spinner("Generating explanations..."):
            # Prepare data
            X = data.drop(['is_fraud', 'transaction_id', 'customer_id', 'timestamp'], axis=1, errors='ignore')
            y = data['is_fraud']
            
            # Sample data for explanation
            sample_data = X.sample(n=sample_size, random_state=42)
            
            # Create explainer
            explainer = FraudExplainer(st.session_state.pipeline, sample_data)
            
            if explanation_type == "Global Feature Importance":
                # Global explanations
                global_explanation = explainer.explain_global(sample_data)
                
                st.subheader("Global Feature Importance")
                
                # Top features
                top_features = global_explanation['global_importance'][:15]
                
                fig = px.bar(x=[f['mean_abs_shap'] for f in top_features], 
                           y=[f['feature'] for f in top_features],
                           orientation='h', title="Top 15 Features by SHAP Importance")
                fig.update_layout(xaxis_title="Mean |SHAP Value|", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.subheader("Feature Importance Details")
                importance_df = pd.DataFrame(top_features)
                st.dataframe(importance_df, use_container_width=True)
            
            elif explanation_type == "Individual Prediction":
                # Individual explanations
                st.subheader("Individual Prediction Explanations")
                
                # Select a sample to explain
                sample_idx = st.selectbox("Select Sample to Explain", range(min(10, len(sample_data))))
                
                # Get explanation
                individual_explanation = explainer.explain_prediction(sample_data.iloc[[sample_idx]])
                
                explanation = individual_explanation['explanations'][0]
                
                # Display prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Fraud Probability", f"{explanation['fraud_probability']:.3f}")
                
                with col2:
                    prediction = "🚨 FRAUD" if explanation['fraud_probability'] > 0.5 else "✅ NORMAL"
                    st.metric("Prediction", prediction)
                
                # Top contributing features
                st.subheader("Top Contributing Features")
                
                top_features = explanation['top_features'][:10]
                
                fig = px.bar(x=[f['shap_value'] for f in top_features], 
                           y=[f['feature'] for f in top_features],
                           orientation='h', title="Feature Contributions (SHAP Values)")
                fig.update_layout(xaxis_title="SHAP Value", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature details table
                st.subheader("Feature Details")
                feature_details = pd.DataFrame(top_features)
                st.dataframe(feature_details, use_container_width=True)
            
            else:
                # Sample analysis
                st.subheader("Sample Analysis")
                
                # Get predictions for sample
                predictions = st.session_state.pipeline.predict_proba(sample_data)
                
                # Create analysis DataFrame
                analysis_df = sample_data.copy()
                analysis_df['fraud_probability'] = predictions
                analysis_df['prediction'] = (predictions >= 0.5).astype(int)
                
                # High-risk transactions
                high_risk = analysis_df[analysis_df['fraud_probability'] > 0.7]
                
                st.write(f"**High-risk transactions (probability > 0.7)**: {len(high_risk)}")
                
                if len(high_risk) > 0:
                    st.subheader("High-Risk Transaction Analysis")
                    
                    # Risk factors analysis
                    risk_factors = []
                    for col in ['is_foreign', 'is_high_risk_country', 'is_weekend', 'is_night']:
                        if col in high_risk.columns:
                            factor_rate = high_risk[col].mean()
                            risk_factors.append({
                                'Risk Factor': col.replace('_', ' ').title(),
                                'Rate in High-Risk': f"{factor_rate:.1%}",
                                'Rate in All Data': f"{analysis_df[col].mean():.1%}"
                            })
                    
                    if risk_factors:
                        risk_df = pd.DataFrame(risk_factors)
                        st.dataframe(risk_df, use_container_width=True)
                    
                    # Amount analysis
                    st.subheader("Amount Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Avg Amount (High-Risk)", f"${high_risk['amount'].mean():.2f}")
                    
                    with col2:
                        st.metric("Avg Amount (All Data)", f"${analysis_df['amount'].mean():.2f}")
                    
                    # Amount distribution
                    fig = px.histogram(analysis_df, x='amount', color='prediction', 
                                     title="Amount Distribution by Prediction",
                                     labels={'prediction': 'Predicted Fraud'})
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
