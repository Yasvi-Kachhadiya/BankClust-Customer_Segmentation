from flask import Flask, render_template, request
import pandas as pd
import pickle
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

app = Flask(__name__)

CLUSTER_LABELS = {
    0: "High Value Customers",
    1: "Low Activity Customers",
    2: "Credit Risk Customers",
    3: "Revolvers",
    4: "Occasional Spenders",
    5: "Transactors"
}

# Load pretrained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Determine feature names from scaler
FEATURES = list(scaler.feature_names_in_)

@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = None
    dist_chart = None
    pie_chart = None
    line_chart = None
    stats = {}
    message = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            try:
                # Read CSV and drop ID if present
                df = pd.read_csv(file)
                if 'CUST_ID' in df.columns:
                    df = df.drop(columns=['CUST_ID'])

                # Select and impute missing
                data = df[FEATURES]
                if data.isnull().any().any():
                    data = data.fillna(data.mean())

                # Scale and predict
                scaled = scaler.transform(data)
                # After predicting clusters
                labels = model.predict(scaled)
                df['Cluster'] = labels
                df['Segment'] = df['Cluster'].map(CLUSTER_LABELS)

                # Show full dataset not just 10 rows (optional: use head if too large)
                table_html = df.to_html(classes='data', index=False)

                # Cluster distribution bar chart
                counts = df['Cluster'].value_counts().reindex(range(model.n_clusters), fill_value=0)
                fig = Figure()
                ax = fig.subplots()
                counts.plot(kind='bar', ax=ax)
                ax.set_title('Cluster Distribution')
                ax.set_xlabel('Cluster Label')
                ax.set_ylabel('Count')
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                dist_chart = base64.b64encode(buf.getvalue()).decode('utf8')

                # Pie chart of distribution
                fig2 = Figure()
                ax2 = fig2.subplots()
                counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
                ax2.set_ylabel('')
                ax2.set_title('Cluster Proportions')
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format='png')
                buf2.seek(0)
                pie_chart = base64.b64encode(buf2.getvalue()).decode('utf8')

                # Compute cluster centers and feature variation
                centers = scaler.inverse_transform(model.cluster_centers_)
                centers_df = pd.DataFrame(centers, columns=FEATURES)
                variation = centers_df.max() - centers_df.min()
                top_feature = variation.idxmax()
                stats['Top Feature'] = top_feature
                stats['Max Variation'] = round(variation[top_feature], 2)

                # Line chart for top feature across clusters
                fig3 = Figure()
                ax3 = fig3.subplots()
                ax3.plot(centers_df.index, centers_df[top_feature], marker='o')
                ax3.set_xticks(centers_df.index)
                ax3.set_xlabel('Cluster Label')
                ax3.set_ylabel(top_feature)
                ax3.set_title(f'{top_feature} Across Clusters')
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format='png')
                buf3.seek(0)
                line_chart = base64.b64encode(buf3.getvalue()).decode('utf8')

            except Exception as e:
                message = f'Error processing file: {e}'
        else:
            message = 'Please upload a valid CSV file.'

    return render_template('index.html',
                           table=table_html,
                           dist_chart=dist_chart,
                           pie_chart=pie_chart,
                           line_chart=line_chart,
                           stats=stats,
                           message=message)

if __name__ == '__main__':
    app.run(debug=True)