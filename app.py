from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from predict_iris import predict_species, validate_input, train_models

app = Flask(__name__)

SPECIES_STYLES = {
    'Setosa': {'bg': '#fce7f3', 'color': '#9d174d'},
    'Versicolor': {'bg': '#e0f2fe', 'color': '#0c4a6e'},
    'Virginica': {'bg': '#fef9c3', 'color': '#92400e'},
}


def get_species_style(species):
    return SPECIES_STYLES.get(species, {'bg': '#e2e8f0', 'color': '#0f172a'})


def encode_plot(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def get_model_accuracies():
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    if not hasattr(predict_species, 'lr_model'):
        predict_species.lr_model, predict_species.dt_model = train_models()

    accuracy_lr = accuracy_score(y_test, predict_species.lr_model.predict(X_test))
    accuracy_dt = accuracy_score(y_test, predict_species.dt_model.predict(X_test))
    return accuracy_lr, accuracy_dt


def generate_pie_chart():
    iris = load_iris()
    species_counts = {}
    for species in iris.target:
        species_name = iris.target_names[species]
        species_counts[species_name] = species_counts.get(species_name, 0) + 1
    
    labels = list(species_counts.keys())
    sizes = list(species_counts.values())
    colors = ['#f472b6', '#38bdf8', '#f59e0b']  # soft pink, blue, yellow
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Iris Species Distribution', fontsize=16)
    plt.tight_layout()
    return encode_plot(fig)


def format_report_without_support(report_str):
    lines = report_str.split('\n')
    result_lines = []
    for line in lines:
        if 'support' in line:
            # remove 'support' from header
            line = line.rsplit(' ', 1)[0].rstrip()
            result_lines.append(line)
        elif line.strip():
            # for lines with content, remove last word if it's a number (support)
            parts = line.split()
            if parts and parts[-1].isdigit():
                line = line.rsplit(' ', 1)[0].rstrip()
                result_lines.append(line)
            else:
                result_lines.append(line.rstrip())
        else:
            result_lines.append(line)
    return '\n'.join(result_lines)


def build_results_charts():
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    if not hasattr(predict_species, 'lr_model'):
        predict_species.lr_model, predict_species.dt_model = train_models()

    y_pred_lr = predict_species.lr_model.predict(X_test)
    y_pred_dt = predict_species.dt_model.predict(X_test)

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)

    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    model_labels = ['Logistic Regression (LR)', 'Decision Tree (DT)']
    accuracies = [accuracy_lr * 100, accuracy_dt * 100]
    bars = ax_acc.bar(range(len(model_labels)), accuracies, color=['#f472b6', '#38bdf8'])
    ax_acc.set_ylim(0, max(accuracies) + 12)
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Model Accuracy Comparison')
    ax_acc.set_xlabel('Model')
    ax_acc.set_xticks(range(len(model_labels)))
    ax_acc.set_xticklabels(model_labels, rotation=0, ha='center')
    ax_acc.tick_params(axis='x', labelsize=10)
    for bar in bars:
        height = bar.get_height()
        ax_acc.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha='center', va='bottom', fontsize=12,
            color='#111827',
            xytext=(0, 8), textcoords='offset points'
        )
    fig_acc.tight_layout()
    acc_chart = encode_plot(fig_acc)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fig_cm_lr, ax_cm_lr = plt.subplots(figsize=(6.5, 6))
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        annot_kws={'size': 14},
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        ax=ax_cm_lr,
    )
    ax_cm_lr.set_title('Logistic Regression Confusion Matrix', fontsize=14)
    ax_cm_lr.set_xlabel('Predicted', fontsize=12)
    ax_cm_lr.set_ylabel('Actual', fontsize=12)
    ax_cm_lr.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax_cm_lr.tick_params(axis='y', labelsize=12)
    fig_cm_lr.tight_layout()
    cm_chart_lr = encode_plot(fig_cm_lr)

    cm_dt = confusion_matrix(y_test, y_pred_dt)
    fig_cm_dt, ax_cm_dt = plt.subplots(figsize=(6.5, 6))
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt='d',
        cmap='Purples',
        cbar=False,
        annot_kws={'size': 14},
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        ax=ax_cm_dt,
    )
    ax_cm_dt.set_title('Decision Tree Confusion Matrix', fontsize=14)
    ax_cm_dt.set_xlabel('Predicted', fontsize=12)
    ax_cm_dt.set_ylabel('Actual', fontsize=12)
    ax_cm_dt.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax_cm_dt.tick_params(axis='y', labelsize=12)
    fig_cm_dt.tight_layout()
    cm_chart_dt = encode_plot(fig_cm_dt)

    report_lr = classification_report(y_test, y_pred_lr, target_names=iris.target_names)
    report_dt = classification_report(y_test, y_pred_dt, target_names=iris.target_names)

    report_lr = format_report_without_support(report_lr)
    report_dt = format_report_without_support(report_dt)

    return acc_chart, cm_chart_lr, cm_chart_dt, report_lr, report_dt, accuracy_lr, accuracy_dt


@app.route('/')
def home():
    return render_template('index.html', title='Iris Bloom', active_page='home')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    error = None
    badge_bg_lr = None
    badge_color_lr = None
    badge_bg_dt = None
    badge_color_dt = None

    if request.method == 'POST':
        try:
            sepal_length = request.form.get('sepal_length', '').strip()
            sepal_width = request.form.get('sepal_width', '').strip()
            petal_length = request.form.get('petal_length', '').strip()
            petal_width = request.form.get('petal_width', '').strip()

            sl, sw, pl, pw = validate_input(sepal_length, sepal_width, petal_length, petal_width)
            lr_species, _ = predict_species(sl, sw, pl, pw, model='lr')
            dt_species, _ = predict_species(sl, sw, pl, pw, model='dt')
            lr_accuracy, dt_accuracy = get_model_accuracies()

            result = {
                'lr_species': lr_species,
                'dt_species': dt_species,
                'lr_accuracy': lr_accuracy * 100,
                'dt_accuracy': dt_accuracy * 100,
                'sepal_length': sl,
                'sepal_width': sw,
                'petal_length': pl,
                'petal_width': pw,
            }
            style_lr = get_species_style(lr_species)
            style_dt = get_species_style(dt_species)
            badge_bg_lr = style_lr['bg']
            badge_color_lr = style_lr['color']
            badge_bg_dt = style_dt['bg']
            badge_color_dt = style_dt['color']
        except ValueError as exc:
            error = str(exc)

    return render_template(
        'predict.html',
        title='Iris Predictor',
        active_page='predict',
        result=result,
        error=error,
        badge_bg_lr=badge_bg_lr,
        badge_color_lr=badge_color_lr,
        badge_bg_dt=badge_bg_dt,
        badge_color_dt=badge_color_dt,
    )


@app.route('/results')
def results():
    acc_chart, cm_chart_lr, cm_chart_dt, report_lr, report_dt, accuracy_lr, accuracy_dt = build_results_charts()
    return render_template(
        'results.html',
        title='Model Comparison',
        active_page='results',
        acc_chart=acc_chart,
        cm_chart_lr=cm_chart_lr,
        cm_chart_dt=cm_chart_dt,
        report_lr=report_lr,
        report_dt=report_dt,
        accuracy_lr=accuracy_lr * 100,
        accuracy_dt=accuracy_dt * 100,
    )


@app.route('/how', endpoint='how')
def how_it_works():
    return render_template('how.html', title='How it Works', active_page='how')


@app.route('/species', endpoint='species')
def species():
    return render_template('species.html', title='Iris Species', active_page='species')


@app.route('/eda', endpoint='eda')
def eda():
    return render_template('eda.html', title='EDA Analysis', active_page='eda')


if __name__ == '__main__':
    print('Starting Iris Classification App on port 5001...')
    app.run(debug=True, port=5001)
