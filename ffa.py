# Общие библиотеки
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools # Для Grid Search
import time      # Для замера времени Grid Search
from collections import Counter # Для анализа стабильности
import inspect     # Для проверки параметров функций
from sklearn.model_selection import LeaveOneOut

# Параметры визуализации (опционально)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.5,
    "font.size": 12
})

# ===============================================
# 1. Предобработка данных
# ===============================================
def preprocess_data(X: np.ndarray, variance_threshold: float = 0.01, correlation_threshold: float = 0.9):
    """
    Предобработка данных для отбора признаков.

    Параметры
    ----------
    X : np.ndarray
        Исходная матрица признаков.
    variance_threshold : float
        Порог удаления признаков с низкой дисперсией.
    correlation_threshold : float
        Порог удаления высококоррелированных признаков.

    Возвращает
    ----------
    X_preprocessed : np.ndarray
        Отфильтрованная матрица признаков.
    kept_indices_original : list
        Индексы оставшихся признаков от ИСХОДНОГО X.
    dropped_corr_indices_relative : list
        Индексы удалённых признаков (корреляция) ОТНОСИТЕЛЬНО признаков ПОСЛЕ фильтрации по дисперсии.
    """
    initial_indices = np.arange(X.shape[1])

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Фильтрация по дисперсии
    selector = VarianceThreshold(variance_threshold)
    try:
        X_var_filtered = selector.fit_transform(X_scaled)
        if X_var_filtered.shape[1] == 0:
            print("Предупреждение: Все признаки удалены фильтром VarianceThreshold.")
            return np.empty((X.shape[0], 0)), [], []
        indices_after_var = initial_indices[selector.get_support(indices=True)]
    except ValueError as e:
         print(f"Ошибка при фильтрации по дисперсии: {e}. Пропускаем этот шаг.")
         X_var_filtered = X_scaled
         indices_after_var = initial_indices

    # Фильтрация по корреляции
    df_var_filtered = pd.DataFrame(X_var_filtered)
    corr_matrix = df_var_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_relative = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    df_preprocessed = df_var_filtered.drop(columns=to_drop_relative)
    kept_cols_relative = df_preprocessed.columns.tolist()

    # Преобразование относительных индексов в индексы исходной матрицы X
    final_kept_indices_original = indices_after_var[kept_cols_relative].tolist()

    dropped_corr_indices_relative = to_drop_relative

    return df_preprocessed.values, final_kept_indices_original, dropped_corr_indices_relative

# ========================================================================
# 2. Предварительная фильтрация признаков (ANOVA)
# ========================================================================
def prefilter_by_anova_alpha(X: np.ndarray, y: np.ndarray, alpha: float = 0.01):
    """
    Отбирает признаки по критерию ANOVA с p-value < alpha.

    Параметры
    ----------
    X : np.ndarray
        Матрица признаков (предполагается, что это X_preprocessed).
    y : np.ndarray
        Вектор меток классов.
    alpha : float, optional
        Уровень значимости для отбора признаков (default=0.01).

    Возвращает
    ----------
    selected_indices_relative : np.ndarray[int]
        Индексы признаков, прошедших фильтрацию (относительно входного X).
    """
    try:
        F, pvals = f_classif(X, y)
        pvals = np.nan_to_num(pvals, nan=1.0) # Замена NaN на 1.0 для безопасного сравнения
        selected_indices_relative = np.where(pvals < alpha)[0]
    except ValueError as e:
        print(f"Предупреждение в ANOVA: {e}. Возможно, все признаки константны. Возвращаем пустой массив.")
        selected_indices_relative = np.array([], dtype=int)
    return selected_indices_relative

# ========================================================================
# 3. Локальный поиск для улучшения решения
# ========================================================================
def local_search_improvement(X: np.ndarray, y: np.ndarray, current_solution: np.ndarray,
                             fitness_func, max_flips: int = 20):
    """
    Улучшает текущий набор признаков через локальный поиск (hill-climbing).
    Пробует инвертировать биты в решении и принимает улучшение, если оно найдено.
    """
    best_solution = current_solution.copy()
    try:
        best_fitness = fitness_func(best_solution)
    except Exception as e:
         print(f"Ошибка при вычислении начального фитнеса в local_search: {e}")
         return current_solution, -np.inf # Возвращаем исходное решение и плохой фитнес

    flips = 0
    improved = True

    while flips < max_flips and improved:
        improved = False
        # Итерация по случайно переставленным индексам для исследования соседей
        indices = np.random.permutation(len(current_solution))
        for idx in indices:
            if len(current_solution) == 0: # Проверка на случай пустого решения
                break
            candidate = best_solution.copy()
            candidate[idx] = 1 - candidate[idx] # Инвертируем бит
            try:
                candidate_fitness = fitness_func(candidate)
                if candidate_fitness > best_fitness:
                    best_solution = candidate.copy()
                    best_fitness = candidate_fitness
                    improved = True
                    flips += 1
                    break # Переходим к следующей итерации локального поиска с новым лучшим решением
            except Exception:
                # Пропускаем кандидата, если фитнес-функция выдает ошибку
                continue
        if not improved: # Если за полный проход по соседям улучшений не найдено
             break
    return best_solution, best_fitness

# ========================================================================
# 4. Функции визуализации
# ========================================================================
def plot_feature_distribution(X: np.ndarray, selected_indices=None, n_features: int = 10, title: str = "Распределение значений признаков"):
    if X.shape[1] == 0:
        print("Нет признаков для отображения распределения.")
        return
    if selected_indices is None:
        indices_to_plot = np.random.choice(X.shape[1], min(n_features, X.shape[1]), replace=False)
    else:
        valid_indices = [idx for idx in selected_indices if idx < X.shape[1]]
        indices_to_plot = valid_indices[:min(n_features, len(valid_indices))]

    if not indices_to_plot:
         print("Нет валидных индексов для отображения.")
         return

    n_plot = len(indices_to_plot)
    plt.figure(figsize=(3 * n_plot, 5))
    for i, idx in enumerate(indices_to_plot):
        plt.subplot(1, n_plot, i + 1)
        plt.hist(X[:, idx], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Признак {idx}")
        plt.xlabel("Значение")
        if i == 0: plt.ylabel("Частота")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correlation_heatmap(X: np.ndarray, title: str = "Тепловая карта корреляций"):
    if X.shape[1] < 2:
        print("Недостаточно признаков для построения тепловой карты корреляций.")
        return
    df = pd.DataFrame(X)
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='viridis', annot=False) # annot=False для больших матриц
    plt.title(title)
    plt.xlabel("Признаки")
    plt.ylabel("Признаки")
    plt.tight_layout()
    plt.show()

def plot_feature_importance_svm(X, y, selected_indices_preproc, kept_indices_prep_array, title="Важность признаков (SVM)"):
    if not selected_indices_preproc: # Проверка на пустой список
        print("Нет выбранных признаков для оценки важности (SVM).")
        return

    valid_indices_preproc = [idx for idx in selected_indices_preproc if idx < X.shape[1]]
    if not valid_indices_preproc:
         print("Нет валидных selected_indices_preproc для оценки важности (SVM).")
         return
    X_sel = X[:, valid_indices_preproc]
    if X_sel.shape[1] == 0:
         print("Нет признаков после фильтрации индексов для оценки важности (SVM).")
         return

    clf = SVC(kernel='linear', random_state=42)
    try:
        clf.fit(X_sel, y)
    except ValueError as e:
        print(f"Ошибка при обучении SVM для важности: {e}")
        return

    coefs = clf.coef_.flatten()
    importance = np.abs(coefs)
    sorted_inner_idx = np.argsort(importance)[::-1] # Индексы отсортированной важности
    sorted_importance = importance[sorted_inner_idx]

    # Маппинг на оригинальные индексы для отображения
    original_indices_labels = ["N/A"] * len(sorted_inner_idx)
    if kept_indices_prep_array.size > 0:
        try:
             # selected_indices_preproc[i] - это индекс в X_preprocessed
             # kept_indices_prep_array[selected_indices_preproc[i]] - это оригинальный индекс
             original_indices_labels = [kept_indices_prep_array[valid_indices_preproc[i]] for i in sorted_inner_idx]
        except IndexError:
             print("Предупреждение: ошибка маппинга индексов при построении важности.")

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(original_indices_labels)), sorted_importance, align='center', color='teal')
    plt.xticks(range(len(original_indices_labels)), original_indices_labels, rotation=90)
    plt.xlabel("Индекс признака (исходный)")
    plt.ylabel("Абсолютное значение коэффициента SVM")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_2d_projection(X: np.ndarray, y: np.ndarray, selected_indices_preproc, method: str = "pca", title: str = "2D Projection of Selected Features"):
    valid_indices = [idx for idx in selected_indices_preproc if idx < X.shape[1]]
    if len(valid_indices) < 2:
        print(f"Недостаточно валидных признаков ({len(valid_indices)}) для 2D-проекции (нужно хотя бы 2).")
        return
    X_sel = X[:, valid_indices]

    if method == "pca":
        n_comp = min(2, X_sel.shape[1]) # Не более 2 компонент и не более числа признаков
        if n_comp < 2: # Если после выбора признаков их меньше 2
            print(f"Для PCA требуется как минимум 2 признака, получено {n_comp}.")
            return
        reducer = PCA(n_components=n_comp, random_state=42)
    else:
        raise ValueError("Метод проекции должен быть 'pca'")

    try:
        X_2d = reducer.fit_transform(X_sel)
    except ValueError as e: # Обработка ошибок, если PCA не может быть применен
        print(f"Ошибка при выполнении {method.upper()}: {e}")
        return

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter, label="Класс")
    plt.xlabel(f"{method.upper()} Компонента 1")
    plt.ylabel(f"{method.upper()} Компонента 2")
    plt.tight_layout()
    plt.show()

def plot_accuracy_comparison(dataset_results, dataset_name):
    """ Отображает bar-plot accuracy для всех методов на одном датасете. """
    methods_ = []
    accuracies = []
    # Фильтруем результаты, чтобы убедиться в наличии ключа 'accuracy' и его корректном типе
    valid_results = {name: res for name, res in dataset_results.items() if 'accuracy' in res and isinstance(res['accuracy'], (int, float))}

    if not valid_results:
        print(f"Нет валидных данных accuracy для построения графика для датасета '{dataset_name}'.")
        return

    sorted_methods = sorted(valid_results.keys()) # Сортируем для консистентного отображения
    for method_name in sorted_methods:
        methods_.append(method_name)
        accuracies.append(valid_results[method_name]["accuracy"])

    plt.figure(figsize=(12, 7))
    bars = plt.bar(methods_, accuracies, color='skyblue', edgecolor='black')
    plt.xlabel("Метод")
    plt.ylabel("Accuracy")
    min_acc = min(accuracies) if accuracies else 0
    max_acc = max(accuracies) if accuracies else 1
    # Динамический диапазон оси Y для лучшей визуализации
    plt.ylim([max(0, min_acc - 0.1 * (max_acc - min_acc) if max_acc > min_acc else min_acc - 0.1 ),
              min(1.0, max_acc + 0.1 * (max_acc - min_acc) if max_acc > min_acc else max_acc + 0.1)])
    for bar in bars: # Добавляем значения на столбцы
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', rotation=0, size=10)

    plt.title(f"Сравнение accuracy для датасета '{dataset_name}'")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_feature_stability(stability_counts, top_n=30, n_runs=None, title="Стабильность выбора признаков"):
    """ Строит гистограмму частоты выбора признаков. """
    if not stability_counts:
        print("Нет данных о стабильности для отображения.")
        return

    sorted_features = stability_counts.most_common(top_n)
    if not sorted_features:
        print("Нет выбранных признаков для отображения стабильности.")
        return

    indices = [item[0] for item in sorted_features]
    counts = [item[1] for item in sorted_features]

    plt.figure(figsize=(12, 7))
    if n_runs: # Отображение в процентах, если известно общее число запусков
        frequencies = [c / n_runs * 100 for c in counts]
        bars = plt.bar(range(len(indices)), frequencies, color='coral', edgecolor='black')
        plt.ylabel(f"Частота выбора (%) из {n_runs} запусков")
        plt.ylim(0, 105) # Устанавливаем лимит оси Y до 105% для наглядности
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center', size=9)
    else: # Отображение в абсолютных значениях
        bars = plt.bar(range(len(indices)), counts, color='coral', edgecolor='black')
        plt.ylabel("Частота выбора (количество запусков)")
        plt.ylim(0, max(counts) * 1.1) # Динамический лимит оси Y
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', size=9)

    plt.xlabel("Индекс признака (исходный)")
    plt.title(f"{title} (Топ-{len(indices)} самых стабильных)")
    plt.xticks(range(len(indices)), indices, rotation=90)
    plt.tight_layout()
    plt.show()

def plot_convergence(fitness_history, n_selected_history, title="Сходимость алгоритма"):
    """ Визуализирует историю фитнеса и количества выбранных признаков. """
    if not fitness_history or not n_selected_history:
        print("Нет данных истории для построения графика сходимости.")
        return

    iterations = range(1, len(fitness_history) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Лучший фитнес', color=color)
    ax1.plot(iterations, fitness_history, color=color, marker='o', linestyle='-', label='Лучший фитнес')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx() # Создание второй оси Y с той же осью X
    color = 'tab:blue'
    ax2.set_ylabel('Количество выбранных признаков', color=color)
    ax2.plot(iterations, n_selected_history, color=color, marker='x', linestyle='--', label='Кол-во признаков')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(title)
    # Объединение легенд для обеих осей
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Корректировка макета для заголовка
    plt.show()


# ========================================================================
# 5. Фитнес-функции
# ========================================================================
def make_svm_fitness(X: np.ndarray, y: np.ndarray, lambda_penalty: float = 0.01, cv=None):
    """ Создаёт фитнес-функцию для SVM с учётом штрафа за количество признаков. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Предупреждение (make_svm_fitness): Входная матрица X не имеет признаков.")
        return lambda firefly: 0.0 # Возвращаем функцию, которая всегда дает 0
    if cv is None: # Установка CV по умолчанию, если не передано
        cv_ = LeaveOneOut() if n_samples < 50 else KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
    else:
        cv_ = cv

    clf = SVC(kernel="linear", C=1.0, random_state=42) # Линейный SVM

    def fitness(firefly: np.ndarray) -> float:
        selected = np.where(firefly == 1)[0] # Индексы выбранных признаков
        n_selected = len(selected)
        if n_selected == 0: return 0.0 # Если нет выбранных признаков
        # Проверка на выход индексов за пределы
        if np.max(selected) >= n_features:
             print(f"Ошибка в fitness: индекс {np.max(selected)} вне границ X ({n_features})")
             return -np.inf # Очень плохой фитнес
        X_selected = X[:, selected]
        # Проверка, что есть хотя бы 2 уникальных образца для CV
        if np.unique(X_selected, axis=0).shape[0] < 2: return 0.0
        try:
            # Оценка accuracy с помощью кросс-валидации
            scores = cross_val_score(clf, X_selected, y, cv=cv_, n_jobs=-1, error_score=0.0)
            acc = np.mean(scores)
        except ValueError: acc = 0.0 # Если CV не удалась
        except Exception as e:
             print(f"Неожиданная ошибка (fitness) при cross_val_score: {e}. Возвращаем 0.")
             acc = 0.0
        # Штраф за количество признаков
        penalty = lambda_penalty * (n_selected / n_features) if n_features > 0 else 0
        return acc - penalty
    return fitness

def make_pca_fitness(X, y, n_components=10, lambda_penalty=0.01, cv=None):
    """ Фитнесс-функция: SVM(linear) после PCA на выбранных признаках. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Предупреждение (make_pca_fitness): Входная матрица X не имеет признаков.")
        return lambda firefly: 0.0
    if cv is None:
        cv_ = LeaveOneOut() if n_samples < 50 else KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
    else:
        cv_ = cv
    clf = SVC(kernel="linear", C=1.0, random_state=42)

    def fitness(firefly):
        selected = np.where(firefly == 1)[0]
        n_selected = len(selected)
        if n_selected == 0: return 0.0
        if np.max(selected) >= n_features:
             print(f"Ошибка в PCA fitness: индекс {np.max(selected)} вне границ X ({n_features})")
             return -np.inf
        X_selected = X[:, selected]
        # Динамическое определение числа компонент для PCA
        dim = min(n_components, X_selected.shape[0], X_selected.shape[1])
        if dim < 1: return 0.0 # Если PCA невозможен
        try:
            pca = PCA(n_components=dim, random_state=42)
            X_pca = pca.fit_transform(X_selected)
            if np.unique(X_pca, axis=0).shape[0] < 2: return 0.0
            scores = cross_val_score(clf, X_pca, y, cv=cv_, n_jobs=-1, error_score=0.0)
            acc = np.mean(scores)
        except ValueError: acc = 0.0
        except Exception as e:
             print(f"Неожиданная ошибка (PCA fitness): {e}. Возвращаем 0.")
             acc = 0.0
        penalty = lambda_penalty * (n_selected / n_features) if n_features > 0 else 0
        return acc - penalty
    return fitness

def make_lasso_fitness(X, y, lambda_penalty=0.01, cv=None):
    """ Фитнесс-функция: LogisticRegression(L1) на выбранных признаках. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Предупреждение (make_lasso_fitness): Входная матрица X не имеет признаков.")
        return lambda firefly: 0.0
    if cv is None:
        cv_ = LeaveOneOut() if n_samples < 50 else KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
    else:
        cv_ = cv
    # Логистическая регрессия с L1 регуляризацией
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=42)

    def fitness(firefly):
        selected = np.where(firefly == 1)[0]
        n_selected = len(selected)
        if n_selected == 0: return 0.0
        if np.max(selected) >= n_features:
             print(f"Ошибка в LASSO fitness: индекс {np.max(selected)} вне границ X ({n_features})")
             return -np.inf
        X_selected = X[:, selected]
        if np.unique(X_selected, axis=0).shape[0] < 2: return 0.0
        try:
            scores = cross_val_score(clf, X_selected, y, cv=cv_, n_jobs=-1, error_score=0.0)
            acc = np.mean(scores)
        except ValueError: acc = 0.0
        except Exception as e:
             print(f"Неожиданная ошибка (LASSO fitness): {e}. Возвращаем 0.")
             acc = 0.0
        penalty = lambda_penalty * (n_selected / n_features) if n_features > 0 else 0
        return acc - penalty
    return fitness

# ========================================================================
# 6. Алгоритмы Firefly
# ========================================================================
def _run_firefly_base(X: np.ndarray, y: np.ndarray, fitness_func, n_fireflies: int, max_iter: int,
                       random_state: int, beta0: float, gamma: float, alpha_param: float,
                       bin_threshold: float, log_interval: int):
    """ Базовая логика итерационного процесса алгоритма Firefly. """
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Инициализация популяции светлячков случайными бинарными векторами
    fireflies = np.random.randint(2, size=(n_fireflies, n_features))
    brightness = np.full(n_fireflies, -np.inf) # Начальная яркость (фитнес)
    for i in range(n_fireflies):
         try:
             brightness[i] = fitness_func(fireflies[i])
         except Exception as e:
             print(f"Ошибка при инициализации brightness для светлячка {i}: {e}")
             # Оставляем -np.inf, если фитнес не может быть вычислен

    best_fitness_overall = np.max(brightness) if np.any(np.isfinite(brightness)) else -np.inf
    fitness_history, n_selected_history = [], [] # Для отслеживания сходимости

    for t in range(max_iter):
        # Уменьшение параметра alpha со временем для усиления эксплуатации
        current_alpha = alpha_param * (1.0 - t / max_iter)
        current_alpha = max(current_alpha, 0.01 * alpha_param) # Обеспечение минимального значения alpha

        for i in range(n_fireflies): # Для каждого светлячка i
            for j in range(n_fireflies): # Сравниваем с каждым светлячком j
                # Движение светлячка i к j, если j ярче
                if np.isfinite(brightness[j]) and brightness[j] > brightness[i]:
                    dist = np.linalg.norm(fireflies[i] - fireflies[j]) # Евклидово расстояние
                    attractiveness = beta0 * np.exp(-gamma * dist**2) # Привлекательность
                    random_step = current_alpha * (np.random.rand(n_features) - 0.5) # Случайный шаг
                    # Обновление позиции светлячка i (в непрерывном пространстве)
                    new_position_cont = fireflies[i] + attractiveness * (fireflies[j] - fireflies[i]) + random_step
                    # Бинаризация новой позиции
                    fireflies[i] = (np.clip(new_position_cont, 0, 1) > bin_threshold).astype(int)

            # Пересчет яркости для обновленного светлячка i
            try:
                 brightness[i] = fitness_func(fireflies[i])
            except Exception:
                 # Если ошибка, устанавливаем плохой фитнес, чтобы избежать выбора этого решения
                 brightness[i] = -np.inf

        # Поиск лучшего решения на текущей итерации
        valid_indices = np.where(np.isfinite(brightness))[0] # Только среди валидных фитнесов
        if len(valid_indices) > 0:
            current_best_idx = valid_indices[np.argmax(brightness[valid_indices])]
            current_best_fitness = brightness[current_best_idx]
            current_n_selected = np.sum(fireflies[current_best_idx])
        else: # Если все фитнесы -inf
            current_best_idx = 0 # Формально берем первого
            current_best_fitness = -np.inf
            current_n_selected = np.sum(fireflies[current_best_idx])

        fitness_history.append(current_best_fitness)
        n_selected_history.append(current_n_selected)

        if current_best_fitness > best_fitness_overall: # Обновление глобально лучшего решения
            best_fitness_overall = current_best_fitness

        if log_interval > 0 and (t + 1) % log_interval == 0:
            best_valid_fitness = np.max(brightness[np.isfinite(brightness)]) if np.any(np.isfinite(brightness)) else -np.inf
            n_sel_best_valid = np.sum(fireflies[np.argmax(brightness)]) if np.any(np.isfinite(brightness)) else 'N/A'
            print(f"Итерация {t+1}/{max_iter}: Лучший валидный fitness = {best_valid_fitness:.4f}, "
                  f"признаков = {n_sel_best_valid}")

    # Выбор лучшего решения по окончании всех итераций
    valid_indices_final = np.where(np.isfinite(brightness))[0]
    if len(valid_indices_final) > 0:
        best_idx_final = valid_indices_final[np.argmax(brightness[valid_indices_final])]
        best_vector = fireflies[best_idx_final].copy()
        best_fitness = brightness[best_idx_final]
    else: # Если не найдено ни одного решения с конечным фитнесом
        best_vector = np.zeros(n_features, dtype=int) # Возвращаем пустой вектор
        best_fitness = -np.inf
        print("Предупреждение: Не найдено ни одного решения с конечным фитнесом в _run_firefly_base.")

    return best_vector, best_fitness, fitness_history, n_selected_history

def firefly_svm_modified(X: np.ndarray, y: np.ndarray, n_fireflies: int = 50, max_iter: int = 50,
                         lambda_penalty: float = 0.01, random_state: int = 42, beta0: float = 1.0,
                         gamma: float = 0.1, alpha_param: float = 1.0, bin_threshold: float = 0.7,
                         cv=None, log_interval: int = 10):
    """ Алгоритм Firefly с SVM в качестве фитнес-функции и последующим локальным поиском. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Ошибка (firefly_svm_modified): Входная матрица X не имеет признаков.")
        return np.array([], dtype=int), 0.0, [], []

    fitness_func = make_svm_fitness(X, y, lambda_penalty=lambda_penalty, cv=cv)

    # Основной этап Firefly
    best_vector_pre_ls, best_fitness_pre_ls, fit_hist, n_sel_hist = _run_firefly_base(
        X, y, fitness_func, n_fireflies, max_iter, random_state, beta0, gamma, alpha_param, bin_threshold, log_interval
    )
    print(f"FA SVM завершен. Лучший fitness до LS: {best_fitness_pre_ls:.4f}, признаков: {np.sum(best_vector_pre_ls)}")

    # Этап локального поиска для улучшения найденного решения
    if np.sum(best_vector_pre_ls) == 0: # Если FA не выбрал признаков, LS не нужен
         print("Локальный поиск пропущен (FA не выбрал признаки).")
         final_vector = best_vector_pre_ls
         final_fitness = best_fitness_pre_ls
    else:
        print("Запуск локального поиска...")
        improved_vector, improved_fitness = local_search_improvement(X, y, best_vector_pre_ls, fitness_func, max_flips=20)
        if improved_fitness > best_fitness_pre_ls:
            print(f"Локальный поиск улучшил решение! Fitness: {improved_fitness:.4f}, признаков: {np.sum(improved_vector)}")
            final_vector = improved_vector
            final_fitness = improved_fitness
        else:
            print("Локальный поиск не улучшил решение.")
            final_vector = best_vector_pre_ls
            final_fitness = best_fitness_pre_ls

    selected_indices = np.where(final_vector == 1)[0]
    # Гарантируем, что индексы не выходят за пределы (хотя это должно контролироваться в _run_firefly_base)
    selected_indices = selected_indices[selected_indices < n_features]
    return selected_indices, final_fitness, fit_hist, n_sel_hist

def firefly_svm_no_local(X: np.ndarray, y: np.ndarray, n_fireflies: int = 50, max_iter: int = 50,
                         lambda_penalty: float = 0.01, random_state: int = 42, beta0: float = 1.0,
                         gamma: float = 0.1, alpha_param: float = 1.0, bin_threshold: float = 0.7,
                         cv=None, log_interval: int = 10):
    """ Алгоритм Firefly с SVM в качестве фитнес-функции БЕЗ локального поиска. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Ошибка (firefly_svm_no_local): Входная матрица X не имеет признаков.")
        return np.array([], dtype=int), 0.0, [], []

    fitness_func = make_svm_fitness(X, y, lambda_penalty=lambda_penalty, cv=cv)

    best_vector, best_fitness, fit_hist, n_sel_hist = _run_firefly_base(
        X, y, fitness_func, n_fireflies, max_iter, random_state, beta0, gamma, alpha_param, bin_threshold, log_interval
    )
    print(f"FA SVM (no local) завершен. Лучший fitness: {best_fitness:.4f}, признаков: {np.sum(best_vector)}")

    selected_indices = np.where(best_vector == 1)[0]
    selected_indices = selected_indices[selected_indices < n_features]
    return selected_indices, best_fitness, fit_hist, n_sel_hist

def firefly_pca_modified(X, y, n_fireflies=50, max_iter=50, lambda_penalty=0.01,
                         random_state=42, beta0=1.0, gamma=0.1, alpha_param=1.0,
                         bin_threshold=0.7, n_components=10, cv=None, log_interval=10):
    """ Алгоритм Firefly с PCA+SVM в качестве фитнес-функции и локальным поиском. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Ошибка (firefly_pca_modified): Входная матрица X не имеет признаков.")
        return np.array([], dtype=int), 0.0, [], []

    fitness_func = make_pca_fitness(X, y, n_components=n_components, lambda_penalty=lambda_penalty, cv=cv)

    best_vector_pre_ls, best_fitness_pre_ls, fit_hist, n_sel_hist = _run_firefly_base(
        X, y, fitness_func, n_fireflies, max_iter, random_state, beta0, gamma, alpha_param, bin_threshold, log_interval
    )
    print(f"FA PCA завершен. Лучший fitness до LS: {best_fitness_pre_ls:.4f}, признаков: {np.sum(best_vector_pre_ls)}")

    if np.sum(best_vector_pre_ls) == 0:
         print("Локальный поиск пропущен (FA не выбрал признаки).")
         final_vector = best_vector_pre_ls
         final_fitness = best_fitness_pre_ls
    else:
        print("Запуск локального поиска (PCA)...")
        improved_vector, improved_fitness = local_search_improvement(X, y, best_vector_pre_ls, fitness_func, max_flips=20)
        if improved_fitness > best_fitness_pre_ls:
            print(f"Локальный поиск улучшил решение (PCA)! Fitness: {improved_fitness:.4f}, признаков: {np.sum(improved_vector)}")
            final_vector = improved_vector
            final_fitness = improved_fitness
        else:
            print("Локальный поиск не улучшил решение (PCA).")
            final_vector = best_vector_pre_ls
            final_fitness = best_fitness_pre_ls

    selected_indices = np.where(final_vector == 1)[0]
    selected_indices = selected_indices[selected_indices < n_features]
    return selected_indices, final_fitness, fit_hist, n_sel_hist

def firefly_lasso_modified(X, y, n_fireflies=50, max_iter=50, lambda_penalty=0.01,
                           random_state=42, beta0=1.0, gamma=0.1, alpha_param=1.0,
                           bin_threshold=0.7, cv=None, log_interval=10):
    """ Алгоритм Firefly с LogReg(L1) в качестве фитнес-функции и локальным поиском. """
    n_samples, n_features = X.shape
    if n_features == 0:
        print("Ошибка (firefly_lasso_modified): Входная матрица X не имеет признаков.")
        return np.array([], dtype=int), 0.0, [], []

    fitness_func = make_lasso_fitness(X, y, lambda_penalty=lambda_penalty, cv=cv)

    best_vector_pre_ls, best_fitness_pre_ls, fit_hist, n_sel_hist = _run_firefly_base(
        X, y, fitness_func, n_fireflies, max_iter, random_state, beta0, gamma, alpha_param, bin_threshold, log_interval
    )
    print(f"FA LASSO завершен. Лучший fitness до LS: {best_fitness_pre_ls:.4f}, признаков: {np.sum(best_vector_pre_ls)}")

    if np.sum(best_vector_pre_ls) == 0:
         print("Локальный поиск пропущен (FA не выбрал признаки).")
         final_vector = best_vector_pre_ls
         final_fitness = best_fitness_pre_ls
    else:
        print("Запуск локального поиска (LASSO)...")
        improved_vector, improved_fitness = local_search_improvement(X, y, best_vector_pre_ls, fitness_func, max_flips=20)
        if improved_fitness > best_fitness_pre_ls:
            print(f"Локальный поиск улучшил решение (LASSO)! Fitness: {improved_fitness:.4f}, признаков: {np.sum(improved_vector)}")
            final_vector = improved_vector
            final_fitness = improved_fitness
        else:
            print("Локальный поиск не улучшил решение (LASSO).")
            final_vector = best_vector_pre_ls
            final_fitness = best_fitness_pre_ls

    selected_indices = np.where(final_vector == 1)[0]
    selected_indices = selected_indices[selected_indices < n_features]
    return selected_indices, final_fitness, fit_hist, n_sel_hist

# ========================================================================
# 7. Функции оценки качества
# ========================================================================
def evaluate_selection_method_extended(X_preprocessed, y, selected_indices_preproc, cv=None):
    """ Расширенная оценка качества модели по выбранному подмножеству признаков. """
    if not isinstance(selected_indices_preproc, (list, np.ndarray)):
        selected_indices_preproc = list(selected_indices_preproc) # Преобразование в список для единообразия

    # Валидация и очистка индексов
    valid_indices = [int(idx) for idx in selected_indices_preproc
                     if isinstance(idx, (int, np.integer)) and 0 <= idx < X_preprocessed.shape[1]]
    valid_indices = sorted(list(set(valid_indices))) # Уникальные и отсортированные
    n_selected = len(valid_indices)

    # Метрики по умолчанию, если оценка невозможна
    default_metrics = {
        "accuracy": 0.0, "roc_auc": 0.0, "f1": 0.0,
        "precision": 0.0, "recall": 0.0, "n_selected": n_selected
    }

    if n_selected == 0: return default_metrics
    if cv is None:
        cv_ = LeaveOneOut() if X_preprocessed.shape[0] < 50 else KFold(n_splits=min(5, X_preprocessed.shape[0]), shuffle=True, random_state=42)
    else:
        cv_ = cv

    X_sel = X_preprocessed[:, valid_indices]
    # Проверка, что после выбора признаков образцы все еще различимы
    if np.unique(X_sel, axis=0).shape[0] < 2:
        print("Предупреждение (evaluate_extended): Выбранные признаки не различают образцы.")
        return default_metrics

    # SVM с возможностью расчета вероятностей для ROC AUC
    clf = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
    scoring = { # Набор метрик для оценки
        "accuracy": "accuracy",
        "roc_auc": "roc_auc_ovr" if len(np.unique(y)) > 2 else "roc_auc", # Для мультиклассовой или бинарной
        "f1": "f1_weighted", # Взвешенный F1 для учета дисбаланса классов
        "precision": "precision_weighted",
        "recall": "recall_weighted"
    }

    try:
        scores = cross_validate(clf, X_sel, y, cv=cv_, scoring=scoring, n_jobs=-1, error_score='raise')
        results = {metric: np.mean(scores[f"test_{metric}"]) for metric in scoring.keys()}
        results["n_selected"] = n_selected
    except ValueError as e:
        print(f"Предупреждение (evaluate_extended) при cross_validate: {e}. Возвращаем нули.")
        results = default_metrics
    except Exception as e:
         print(f"Неожиданная ошибка (evaluate_extended) при cross_validate: {e}. Возвращаем нули.")
         results = default_metrics
    return results

def evaluate_selection_method(X_preprocessed, y, selected_indices_preproc, cv=None):
    """ Оценивает только accuracy модели по выбранному подмножеству признаков. """
    if not isinstance(selected_indices_preproc, (list, np.ndarray)):
        selected_indices_preproc = list(selected_indices_preproc)
    valid_indices = [int(idx) for idx in selected_indices_preproc
                     if isinstance(idx, (int, np.integer)) and 0 <= idx < X_preprocessed.shape[1]]
    valid_indices = sorted(list(set(valid_indices)))

    if len(valid_indices) == 0: return 0.0
    if cv is None:
        cv_ = LeaveOneOut() if X_preprocessed.shape[0] < 50 else KFold(n_splits=min(5, X_preprocessed.shape[0]), shuffle=True, random_state=42)
    else:
        cv_ = cv

    X_sel = X_preprocessed[:, valid_indices]
    if np.unique(X_sel, axis=0).shape[0] < 2: return 0.0

    clf = SVC(kernel='linear', C=1.0, random_state=42)
    try:
        scores = cross_val_score(clf, X_sel, y, cv=cv_, scoring="accuracy", n_jobs=-1, error_score=0.0)
        return np.mean(scores)
    except ValueError: return 0.0
    except Exception as e:
        print(f"Неожиданная ошибка (evaluate_accuracy) при cross_val_score: {e}. Возвращаем 0.")
        return 0.0

# ========================================================================
# 8. Функция Grid Search
# ========================================================================
def run_grid_search_firefly(X_filtered_anova, y, anova_map_to_preprocessed, X_preprocessed,
                            param_grid, base_firefly_params, cv_eval,
                            firefly_function=firefly_svm_modified):
    """ Выполняет Grid Search для алгоритма Firefly для подбора оптимальных гиперпараметров. """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values)) # Все комбинации параметров

    results = []
    best_accuracy = -1.0
    best_params = None
    best_n_selected_preproc = -1 # Количество выбранных признаков относительно X_preprocessed
    best_indices_preproc = []  # Индексы выбранных признаков относительно X_preprocessed

    if X_filtered_anova.shape[1] == 0:
        print("Ошибка (Grid Search): X_filtered_anova не содержит признаков. Grid Search невозможен.")
        return {}, -1.0, -1, [], pd.DataFrame() # Возврат пустых результатов

    print(f"Начинаем Grid Search ({firefly_function.__name__}). Всего комбинаций: {len(all_combinations)}")
    start_time_grid = time.time()

    for i, combo in enumerate(all_combinations):
        current_params = dict(zip(param_names, combo))
        print(f"\n[Grid Search {i+1}/{len(all_combinations)}] Параметры: {current_params}")

        firefly_params = {**base_firefly_params, **current_params}
        # Отбираем только те параметры, которые принимает конкретная FA функция
        fa_only_params = {k: v for k, v in firefly_params.items() if k in inspect.signature(firefly_function).parameters}
        fa_only_params['log_interval'] = 0 # Подавление вывода логов из FA во время GS

        start_time_run = time.time()
        try:
            # Запуск FA на данных после ANOVA фильтрации
            selected_indices_filtered, final_fitness, _, _ = firefly_function(
                X_filtered_anova, y, **fa_only_params
            )

            # Маппинг отобранных индексов (отн. X_filtered_anova) на индексы отн. X_preprocessed
            current_indices_preproc = []
            if len(selected_indices_filtered) > 0:
                 if np.max(selected_indices_filtered) < len(anova_map_to_preprocessed):
                     current_indices_preproc = anova_map_to_preprocessed[selected_indices_filtered]
                     current_indices_preproc = sorted(list(set(current_indices_preproc))) # Уникальные и отсортированные
                 else:
                      print("  Предупреждение: некорректные индексы из FA при маппинге в Grid Search.")

            n_selected = len(current_indices_preproc)

            # Оценка точности на X_preprocessed с использованием cv_eval
            accuracy = evaluate_selection_method(
                X_preprocessed, y, current_indices_preproc, cv=cv_eval
            )

            run_time = time.time() - start_time_run
            print(f"  Результат: {n_selected} признаков (отн. X_prep), Accuracy={accuracy:.4f}, Fitness={final_fitness:.4f} (Время: {run_time:.2f}с)")

            results.append({
                **current_params,
                'n_selected_preproc': n_selected,
                'accuracy': accuracy,
                'final_fitness': final_fitness,
                'runtime_seconds': run_time
            })

            # Обновление лучшего результата: приоритет accuracy, затем меньшее число признаков
            if accuracy > best_accuracy + 1e-9: # Используем небольшой эпсилон для сравнения float
                best_accuracy = accuracy
                best_params = current_params
                best_n_selected_preproc = n_selected
                best_indices_preproc = current_indices_preproc
            elif abs(accuracy - best_accuracy) < 1e-9 and n_selected >= 0 and n_selected < best_n_selected_preproc :
                 best_params = current_params
                 best_n_selected_preproc = n_selected
                 best_indices_preproc = current_indices_preproc

        except Exception as e:
            run_time = time.time() - start_time_run
            print(f"  Ошибка при выполнении {firefly_function.__name__} с параметрами {current_params}: {e}")
            results.append({
                **current_params, 'n_selected_preproc': -1, 'accuracy': -1, 'final_fitness': -1,
                'runtime_seconds': run_time, 'error': str(e)
            })

    total_time_grid = time.time() - start_time_grid
    print(f"\nGrid Search ({firefly_function.__name__}) завершен. Общее время: {total_time_grid:.2f}с")

    if best_params:
        print(f"Лучшие параметры: {best_params}")
        n_splits_eval = cv_eval.get_n_splits(X_preprocessed) if hasattr(cv_eval, 'get_n_splits') else 'N/A'
        print(f"Лучшая Accuracy (на {n_splits_eval}-Fold CV): {best_accuracy:.4f} с {best_n_selected_preproc} признаками (отн. X_prep).")
    else:
        print("Не удалось найти рабочий набор параметров в Grid Search.")

    results_df = pd.DataFrame(results)
    # Сортировка результатов для удобства анализа
    if 'accuracy' in results_df.columns and 'n_selected_preproc' in results_df.columns:
         results_df['accuracy_sort'] = results_df['accuracy'].replace(-1, -np.inf) # Для корректной сортировки ошибок
         results_df = results_df.sort_values(by=['accuracy_sort', 'n_selected_preproc'], ascending=[False, True]).drop(columns=['accuracy_sort'])

    return best_params, best_accuracy, best_n_selected_preproc, best_indices_preproc, results_df

# ========================================================================
# 9. Анализ стабильности
# ========================================================================
def analyze_feature_stability(
    firefly_function,
    X_filtered_anova, y,
    anova_map_to_preprocessed, # Карта: индексы в X_filtered_anova -> индексы в X_preprocessed
    kept_indices_original,     # Список оригинальных индексов, соответствующих X_preprocessed
    X_preprocessed,            # Матрица X_preprocessed (для оценки)
    fa_params,                 # Словарь с ЛУЧШИМИ параметрами для FA из Grid Search
    n_runs: int = 30,
    base_seed: int = 47,
    cv_eval=LeaveOneOut()      # CV для оценки каждого запуска
    ):
    """
    Выполняет многократные запуски FA для оценки стабильности выбора признаков.
    Возвращает счетчик частоты выбора оригинальных индексов признаков и средние метрики.
    """
    print(f"\n--- Анализ стабильности ({firefly_function.__name__}) для {n_runs} запусков ---")
    start_time_stability = time.time()

    all_selected_indices_original = [] # Список списков оригинальных индексов для каждого запуска
    all_run_results = [] # Подробные результаты каждого запуска
    fa_params_run = fa_params.copy()
    # Удаляем параметры, которые будут меняться в цикле (random_state) или не нужны (log_interval)
    fa_params_run.pop('log_interval', None)
    fa_params_run.pop('random_state', None)
    # Установка значений по умолчанию, если они не были определены в fa_params
    fa_params_run.setdefault('n_fireflies', 50) # Типичное значение из статьи
    fa_params_run.setdefault('max_iter', 50)    # Типичное значение из статьи
    fa_params_run.setdefault('beta0', 1.0)
    fa_params_run.setdefault('bin_threshold', 0.7)

    kept_indices_original_array = np.array(kept_indices_original) # Для быстрой индексации

    for i in range(n_runs):
        current_seed = base_seed + i # Новый seed для каждого запуска
        print(f"  Запуск {i+1}/{n_runs} (seed={current_seed})...", end="")
        run_start_time = time.time()

        try:
            current_fa_params = {**fa_params_run, 'random_state': current_seed}
            # Отбираем только параметры, которые принимает текущая FA функция
            allowed_params = {
                k: v for k, v in current_fa_params.items()
                if k in inspect.signature(firefly_function).parameters
            }
            allowed_params['cv'] = cv_eval # Передаем CV для фитнес-функции

            # Запуск FA на X_filtered_anova
            selected_rel, final_fitness, _, _ = firefly_function(
                X_filtered_anova, y,
                **allowed_params,
                log_interval=0 # Подавление логов внутри FA
            )
            # Маппинг отобранных индексов: filtered_anova -> preprocessed -> original
            indices_preproc_run = []
            original_indices_run = []
            if len(selected_rel) > 0: # Если FA что-то выбрал
                 if np.max(selected_rel) < len(anova_map_to_preprocessed):
                     indices_preproc_run = anova_map_to_preprocessed[selected_rel]
                     indices_preproc_run = sorted(list(set(indices_preproc_run)))

                     if kept_indices_original_array.size > 0:
                         valid_relative_indices = [idx for idx in indices_preproc_run if idx < len(kept_indices_original_array)]
                         if valid_relative_indices:
                             original_indices_run = kept_indices_original_array[valid_relative_indices].tolist()
                             original_indices_run.sort()
                 else:
                      print(f" Предупреждение: некорректные индексы ({np.max(selected_rel)}) из FA в анализе стабильности.", end="")

            n_selected_original = len(original_indices_run)
            all_selected_indices_original.append(original_indices_run)

            # Оценка производительности на X_preprocessed с помощью cv_eval
            metrics = {'accuracy': -1, 'n_selected': len(indices_preproc_run), 'final_fitness': final_fitness}
            if cv_eval: # Если CV предоставлен
                 metrics = evaluate_selection_method_extended(
                     X_preprocessed, y, indices_preproc_run, cv=cv_eval
                 )
                 metrics['final_fitness'] = final_fitness # Добавляем фитнес из FA
                 metrics['n_selected_original'] = n_selected_original # Кол-во оригинальных признаков

            run_time = time.time() - run_start_time
            acc_val = metrics.get('accuracy', 'N/A')
            acc_str = f"{acc_val:.4f}" if isinstance(acc_val, float) else str(acc_val)
            print(f" Признаков orig: {n_selected_original}, Acc: {acc_str}, Fit: {final_fitness:.4f} ({run_time:.1f}с)")

            run_data = { # Сохранение результатов этого запуска
                'run': i + 1, 'seed': current_seed, 'runtime': run_time,
                'n_selected_original': n_selected_original,
                'selected_indices_original': original_indices_run,
                'accuracy': metrics.get('accuracy'),
                'roc_auc': metrics.get('roc_auc'),
                'f1': metrics.get('f1'),
                'final_fitness': metrics.get('final_fitness'),
                'n_selected_preproc': len(indices_preproc_run) # Кол-во признаков отн. X_preprocessed
            }
            all_run_results.append(run_data)

        except Exception as e: # Обработка ошибок во время запуска
            run_time = time.time() - run_start_time
            print(f" ОШИБКА: {e} ({run_time:.1f}с)")
            all_selected_indices_original.append([]) # Добавляем пустой список при ошибке
            all_run_results.append({
                'run': i + 1, 'seed': current_seed, 'error': str(e), 'runtime': run_time,
                'n_selected_original': -1, 'accuracy': -1, 'final_fitness': -np.inf
            })

    # Агрегация результатов стабильности: подсчет частоты выбора каждого оригинального признака
    flat_list = [item for sublist in all_selected_indices_original for item in sublist]
    stability_counts_original = Counter(flat_list)

    # Расчет средних метрик по всем успешным запускам
    valid_results = [r for r in all_run_results if 'error' not in r and r.get('accuracy') is not None and r['accuracy'] != -1]
    avg_metrics = {}
    if valid_results:
        keys_to_average = ['n_selected_original', 'n_selected_preproc', 'accuracy', 'roc_auc', 'f1', 'final_fitness', 'runtime']
        for key in keys_to_average:
            values = [r[key] for r in valid_results if r.get(key) is not None] # Учитываем возможные None
            avg_metrics[f"avg_{key}"] = np.mean(values) if values else 'N/A'
        avg_metrics["n_valid_runs"] = len(valid_results)
    else:
         avg_metrics["error"] = "No valid runs completed for stability analysis"
         avg_metrics["n_valid_runs"] = 0

    total_stability_time = time.time() - start_time_stability
    print(f"--- Анализ стабильности завершен ({total_stability_time:.2f}с) ---")

    return stability_counts_original, avg_metrics, all_run_results


# ========================================================================
# Основной блок исполнения
# ========================================================================
if __name__ == "__main__":

    # --- Загрузка датасетов ---
    try:
        data_dict_lung = sio.loadmat("lung_GSE68571.mat")
        X_lung = data_dict_lung["X"]
        y_lung = data_dict_lung["y"].ravel().astype(int) # Гарантируем целочисленные метки
    except FileNotFoundError:
        print("Файл lung_GSE68571.mat не найден.")
        X_lung, y_lung = None, None
    except Exception as e:
        print(f"Ошибка при загрузке lung_GSE68571.mat: {e}")
        X_lung, y_lung = None, None

    try:
        data_dict_colon = sio.loadmat("colonU.mat")
        X_colon = data_dict_colon["X"]
        y_colon = data_dict_colon["Y"].ravel().astype(int)
        y_colon[y_colon == -1] = 0 # Преобразование меток -1/1 в 0/1
    except FileNotFoundError:
        print("Файл colonU.mat не найден.")
        X_colon, y_colon = None, None
    except Exception as e:
        print(f"Ошибка при загрузке colonU.mat: {e}")
        X_colon, y_colon = None, None

    try:
        data_dict_leukemia = sio.loadmat("leukemia1.mat")
        X_leukemia = data_dict_leukemia["X"]
        y_leukemia = data_dict_leukemia["y"].ravel().astype(int)
        y_leukemia[y_leukemia == 2] = 0 # Преобразование меток 1/2 в 1/0 (или 0/1 если y_leukemia[y_leukemia == 1] = 1)
                                        # Если метки 1 и 2, то 1->1, 2->0. Если нужно 1->0, 2->1, то:
                                        # y_leukemia_temp = y_leukemia.copy()
                                        # y_leukemia[y_leukemia_temp == 1] = 0
                                        # y_leukemia[y_leukemia_temp == 2] = 1
    except FileNotFoundError:
        print("Файл leukemia1.mat не найден.")
        X_leukemia, y_leukemia = None, None
    except Exception as e:
        print(f"Ошибка при загрузке leukemia1.mat: {e}")
        X_leukemia, y_leukemia = None, None

    datasets = {}
    if X_lung is not None: datasets["lung"] = (X_lung, y_lung)
    if X_colon is not None: datasets["colon"] = (X_colon, y_colon)
    if X_leukemia is not None: datasets["leukemia1"] = (X_leukemia, y_leukemia)

    if not datasets:
        print("Нет успешно загруженных датасетов. Завершение работы.")
        exit()

    # --- Параметры Grid Search (4*4*3 = 48 комбинаций, как в Приложении 1) ---
    param_grid_to_search = {
        'lambda_penalty': [0.001, 0.01, 0.05, 0.1],
        'gamma': [0.1, 0.5, 1.0, 5.0],
        'alpha_param': [0.5, 1.0, 1.5]
    }
    # Базовые параметры FA для Grid Search
    base_firefly_config_gs = {
        'n_fireflies': 50, # Согласно тексту (стр. 25, 26)
        'max_iter': 30,    # Согласно тексту (стр. 25)
        'random_state': 42,
        'beta0': 1.0,      # Согласно тексту (стр. 25)
        'bin_threshold': 0.7, # Согласно тексту (стр. 25)
        'cv': LeaveOneOut() # CV для фитнес-функции внутри FA во время GS
    }
    # Базовые параметры FA для финальных запусков и анализа стабильности
    base_firefly_config_final = {
        'n_fireflies':  50, # Согласно тексту (стр. 25, 26)
        'max_iter': 50,     # Согласно тексту (стр. 25)
        'random_state': 48, # Будет перезаписан в цикле стабильности
        'beta0': 1.0,
        'bin_threshold': 0.7,
        'cv': LeaveOneOut() # CV для фитнес-функции внутри FA
    }

    # CV для финальной оценки accuracy методов (используется в evaluate_selection_method/extended)
    final_evaluation_cv_grid = LeaveOneOut() # Для оценки каждой комбинации GS
    final_evaluation_cv_main = LeaveOneOut() # Для итоговой оценки методов

    N_STABILITY_RUNS = 20 # Согласно тексту (стр. 26, 28, 31, 33)

    overall_results = {} # Хранение всех результатов

    # --- Основной цикл по датасетам ---
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*15} Обработка датасета: {dataset_name} {'='*15}")
        dataset_results = {}

        # 1. Предобработка данных
        print("\n--- Шаг 1: Предобработка ---")
        X_preprocessed, kept_indices_original, _ = preprocess_data(
            X, variance_threshold=0.01, correlation_threshold=0.9
        )
        if X_preprocessed.shape[1] == 0:
             print(f"Ошибка: После предобработки не осталось признаков для {dataset_name}. Пропуск.")
             overall_results[dataset_name] = {"error": "No features after preprocessing"}
             continue
        print(f"Исходная форма X: {X.shape}")
        print(f"Форма после предобработки (Var + Corr): {X_preprocessed.shape}")
        print(f"Количество оставшихся признаков (оригинальные индексы): {len(kept_indices_original)}")
        kept_indices_original_array = np.array(kept_indices_original) # Для удобного маппинга

        # 2. ANOVA фильтрация (Предварительная фильтрация)
        print("\n--- Шаг 2: ANOVA фильтрация ---")
        anova_sel_indices_relative = prefilter_by_anova_alpha(X_preprocessed, y, alpha=0.01)
        if len(anova_sel_indices_relative) == 0:
            print(f"Предупреждение: ANOVA не отобрала ни одного признака для {dataset_name}.")
            X_filtered_anova = np.empty((X_preprocessed.shape[0], 0))
            anova_map_to_preprocessed = np.array([], dtype=int) # Пустая карта индексов
        else:
            X_filtered_anova = X_preprocessed[:, anova_sel_indices_relative]
            print(f"Отобрано по ANOVA (индексы относительно X_preprocessed): {len(anova_sel_indices_relative)} признаков")
            print(f"Форма данных для Firefly (X_filtered_anova): {X_filtered_anova.shape}")
            # Карта: индекс в X_filtered_anova -> индекс в X_preprocessed
            anova_map_to_preprocessed = anova_sel_indices_relative

        # 3. Базовые методы сравнения (Baselines)
        print("\n--- Шаг 3: Базовые сравнения (Baselines) ---")
        # Baseline 1: Все признаки после предобработки
        print("Baseline 1: Все признаки (после предобработки)...")
        indices_all_preproc = list(range(X_preprocessed.shape[1]))
        baseline_all_metrics = evaluate_selection_method_extended(
            X_preprocessed, y, indices_all_preproc, cv=final_evaluation_cv_main
        )
        print(f"  Кол-во признаков: {baseline_all_metrics['n_selected']}, Accuracy: {baseline_all_metrics['accuracy']:.4f}, ROC-AUC: {baseline_all_metrics['roc_auc']:.4f}, F1: {baseline_all_metrics['f1']:.4f}")
        dataset_results["Baseline: All Features (Prep)"] = {
            "accuracy": baseline_all_metrics['accuracy'], "n_selected": baseline_all_metrics['n_selected'],
            "selected_indices_preproc": indices_all_preproc,
            "selected_indices_original": kept_indices_original, # Все оригинальные после предобработки
            "extended_metrics": baseline_all_metrics, "fitness_history": [], "n_selected_history": []
        }

        # Baseline 2: Только признаки, отобранные ANOVA
        print("Baseline 2: Признаки после ANOVA...")
        baseline_anova_metrics = evaluate_selection_method_extended(
            X_preprocessed, y, anova_map_to_preprocessed, cv=final_evaluation_cv_main
        )
        original_indices_anova = [] # Маппинг индексов ANOVA на оригинальные
        if len(anova_map_to_preprocessed) > 0 and kept_indices_original_array.size > 0:
            try:
                valid_anova_indices = [idx for idx in anova_map_to_preprocessed if idx < len(kept_indices_original_array)]
                if valid_anova_indices:
                   original_indices_anova = kept_indices_original_array[valid_anova_indices].tolist()
                   original_indices_anova.sort()
            except IndexError: print(" Ошибка маппинга индексов ANOVA на оригинальные для Baseline")
        print(f"  Кол-во признаков: {baseline_anova_metrics['n_selected']}, Accuracy: {baseline_anova_metrics['accuracy']:.4f}, ROC-AUC: {baseline_anova_metrics['roc_auc']:.4f}, F1: {baseline_anova_metrics['f1']:.4f}")
        dataset_results["Baseline: ANOVA Only"] = {
            "accuracy": baseline_anova_metrics['accuracy'], "n_selected": baseline_anova_metrics['n_selected'],
            "selected_indices_preproc": anova_map_to_preprocessed.tolist(),
            "selected_indices_original": original_indices_anova,
            "extended_metrics": baseline_anova_metrics, "fitness_history": [], "n_selected_history": []
        }

        # 4. Grid Search для основного метода BFA+LS+SVM (firefly_svm_modified)
        print("\n--- Шаг 4: Grid Search для BFA+LS+SVM (firefly_svm_modified) ---")
        best_params_gs = {} # Лучшие параметры, найденные GS
        best_indices_preproc_gs = [] # Соответствующие им индексы отн. X_preprocessed

        if X_filtered_anova.shape[1] > 0: # Если есть на чем запускать FA
            best_params_gs, best_acc_gs, n_sel_gs, best_indices_preproc_gs, results_df_gs = run_grid_search_firefly(
                X_filtered_anova=X_filtered_anova, y=y,
                anova_map_to_preprocessed=anova_map_to_preprocessed,
                X_preprocessed=X_preprocessed,
                param_grid=param_grid_to_search,
                base_firefly_params=base_firefly_config_gs,
                cv_eval=final_evaluation_cv_grid, # CV для оценки каждой комбинации GS
                firefly_function=firefly_svm_modified
            )
            if best_params_gs:
                 print("\nТоп 5 комбинаций из Grid Search (BFA+LS+SVM):")
                 # Вывод релевантных колонок из результатов GS
                 print(results_df_gs[['lambda_penalty', 'gamma', 'alpha_param', 'n_selected_preproc', 'accuracy', 'final_fitness']].head())
            else:
                 print("Grid Search не нашел подходящих параметров для BFA+LS+SVM.")
        else:
            print("Grid Search пропущен, т.к. нет признаков после ANOVA.")

        # 5. Анализ стабильности для лучшей конфигурации BFA+LS+SVM
        print(f"\n--- Шаг 5: Анализ стабильности для BFA+LS+SVM ({N_STABILITY_RUNS} запусков) ---")
        stability_results_svm = {}
        stability_counts_original_svm = Counter()

        if best_params_gs and X_filtered_anova.shape[1] > 0: # Если GS был успешен
            print(f"Используемые параметры FA для анализа стабильности: {best_params_gs}")
            stability_counts_original_svm, avg_metrics_svm, all_runs_svm = analyze_feature_stability(
                firefly_function=firefly_svm_modified, # Основной метод
                X_filtered_anova=X_filtered_anova, y=y,
                anova_map_to_preprocessed=anova_map_to_preprocessed,
                kept_indices_original=kept_indices_original,
                X_preprocessed=X_preprocessed,
                fa_params=best_params_gs, # Параметры из GS
                n_runs=N_STABILITY_RUNS,
                base_seed=42, # Можно использовать другой seed для серии стабильности
                cv_eval=final_evaluation_cv_main # CV для оценки каждого запуска в анализе стабильности
            )
            stability_results_svm = {
                "counts_original": stability_counts_original_svm,
                "avg_metrics": avg_metrics_svm,
                "all_runs": all_runs_svm
            }
            print("\nРезультаты анализа стабильности (BFA+LS+SVM):")
            if avg_metrics_svm.get("n_valid_runs", 0) > 0:
                print(f"  Кол-во успешных запусков: {avg_metrics_svm['n_valid_runs']}/{N_STABILITY_RUNS}")
                print(f"  Среднее Accuracy ({final_evaluation_cv_main.get_n_splits(X_preprocessed) if hasattr(final_evaluation_cv_main, 'get_n_splits') else 'LOO'}-Fold CV): {avg_metrics_svm.get('avg_accuracy', 'N/A'):.4f}")
                print(f"  Среднее ROC AUC: {avg_metrics_svm.get('avg_roc_auc', 'N/A'):.4f}")
                print(f"  Средний F1: {avg_metrics_svm.get('avg_f1', 'N/A'):.4f}")
                print(f"  Среднее кол-во признаков (ориг.): {avg_metrics_svm.get('avg_n_selected_original', 'N/A'):.2f}")
                print(f"  Среднее кол-во признаков (отн. X_prep): {avg_metrics_svm.get('avg_n_selected_preproc', 'N/A'):.2f}")
                print(f"  Средний Fitness: {avg_metrics_svm.get('avg_final_fitness', 'N/A'):.4f}")
                print(f"  Среднее время запуска: {avg_metrics_svm.get('avg_runtime', 'N/A'):.2f}с")
                top_stable = stability_counts_original_svm.most_common(15)
                print(f"\n  Топ-{len(top_stable)} самых стабильных признаков (ориг. индекс: кол-во выборов из {avg_metrics_svm['n_valid_runs']}):")
                for idx, count in top_stable: print(f"    Признак {idx}: {count} раз")
                plot_feature_stability(stability_counts_original_svm, top_n=30, n_runs=avg_metrics_svm['n_valid_runs'],
                                       title=f"Стабильность признаков BFA+LS+SVM ({dataset_name})")
            else: print("  Анализ стабильности: не удалось получить результаты ни одного запуска.")

            # Сохраняем результат лучшего единичного запуска из Grid Search для BFA+LS+SVM
            original_indices_gs_best = []
            if best_indices_preproc_gs and kept_indices_original_array.size > 0:
                 try:
                     valid_gs_indices = [idx for idx in best_indices_preproc_gs if idx < len(kept_indices_original_array)]
                     if valid_gs_indices:
                        original_indices_gs_best = kept_indices_original_array[valid_gs_indices].tolist()
                        original_indices_gs_best.sort()
                 except IndexError: print(" Ошибка маппинга индексов лучшего из GS (BFA+LS+SVM) на оригинальные")
            metrics_gs_best = evaluate_selection_method_extended(X_preprocessed, y, best_indices_preproc_gs, cv=final_evaluation_cv_main)
            dataset_results["BFA+LS+SVM (GS_Best_1_run)"] = { # Имя метода для лучшего запуска из GS
                 "accuracy": metrics_gs_best['accuracy'],
                 "n_selected": metrics_gs_best['n_selected'],
                 "selected_indices_preproc": best_indices_preproc_gs,
                 "selected_indices_original": original_indices_gs_best,
                 "extended_metrics": metrics_gs_best,
                 "best_params": best_params_gs, # Сохраняем параметры, с которыми он был получен
                 "fitness_history": [], "n_selected_history": [] # Истории сходимости для GS нет, она для одиночных запусков
             }
        else: # Если GS не удался или нет признаков после ANOVA
            print("\n--- Шаг 5: Анализ стабильности для BFA+LS+SVM пропущен ---")
            if "BFA+LS+SVM (GS_Best_1_run)" not in dataset_results: # Гарантируем наличие ключа
                 dataset_results["BFA+LS+SVM (GS_Best_1_run)"] = {
                     "accuracy": 0.0, "n_selected": 0, "selected_indices_preproc": [], "selected_indices_original": [],
                     "extended_metrics": evaluate_selection_method_extended(X_preprocessed, y, []), "best_params": {},
                     "fitness_history": [], "n_selected_history": []
                 }

        # 6. Запуск остальных вариантов FA (один раз, с параметрами от GS для BFA+LS+SVM)
        print("\n--- Шаг 6: Запуск других методов Firefly (1 раз, с лучшими параметрами от BFA+LS+SVM GS) ---")
        final_run_params = base_firefly_config_final.copy()
        if best_params_gs: # Если GS нашел параметры
            final_run_params.update(best_params_gs)
            print(f"Используем параметры из Grid Search BFA+LS+SVM: {best_params_gs}")
        else: # Используем дефолтные, если GS не удался
            print("Используем параметры по умолчанию (Grid Search для BFA+LS+SVM не удался).")
            # Установка дефолтных значений для lambda_penalty, gamma, alpha_param, если их нет
            final_run_params.setdefault('lambda_penalty', 0.01)
            final_run_params.setdefault('gamma', 1.0)
            final_run_params.setdefault('alpha_param', 1.0)
        final_run_params['random_state'] = 42 # Фиксированный seed для этих одиночных запусков

        methods_final_run = { # Методы и их функции
            "BFA-NOLS (FA_SVM_no_LS)": firefly_svm_no_local,
            "BFA-LS-PCA (FA_PCA_SVM)": firefly_pca_modified,
            "BFA-LS-LASSO (FA_LASSO)": firefly_lasso_modified,
        }

        if X_filtered_anova.shape[1] > 0: # Если есть на чем запускать
            for method_name, method_func in methods_final_run.items():
                print(f"\nПрименяем метод: {method_name}")
                current_method_params = final_run_params.copy()
                if method_name == "BFA-LS-PCA (FA_PCA_SVM)": # Специфичный параметр для PCA
                    current_method_params['n_components'] = min(10, X_filtered_anova.shape[1], X_filtered_anova.shape[0])
                allowed_params = {k: v for k, v in current_method_params.items() if k in inspect.signature(method_func).parameters}
                allowed_params['log_interval'] = 10 # Включаем логи для финальных одиночных запусков

                try:
                    selected_rel, final_fitness, fit_hist, n_sel_hist = method_func(X_filtered_anova, y, **allowed_params)
                    # Маппинг индексов: filtered_anova -> preprocessed -> original
                    indices_preproc_final = []
                    original_indices_final = []
                    if len(selected_rel) > 0:
                         if np.max(selected_rel) < len(anova_map_to_preprocessed):
                             indices_preproc_final = anova_map_to_preprocessed[selected_rel]
                             indices_preproc_final = sorted(list(set(indices_preproc_final)))
                             if kept_indices_original_array.size > 0:
                                 valid_relative_indices = [idx for idx in indices_preproc_final if idx < len(kept_indices_original_array)]
                                 if valid_relative_indices:
                                     original_indices_final = kept_indices_original_array[valid_relative_indices].tolist()
                                     original_indices_final.sort()
                         else:
                              print("  Предупреждение: некорректные индексы из FA при маппинге.")
                    metrics = evaluate_selection_method_extended(
                        X_preprocessed, y, indices_preproc_final, cv=final_evaluation_cv_main
                    )
                    print(f"  Выбрано признаков (ориг.): {len(original_indices_final)}")
                    print(f"  Оценка Accuracy ({final_evaluation_cv_main.get_n_splits(X_preprocessed) if hasattr(final_evaluation_cv_main, 'get_n_splits') else 'LOO'}-Fold CV): {metrics['accuracy']:.4f}")
                    print(f"  Итоговый fitness (из FA): {final_fitness:.4f}")
                    print(f"  Расширенная оценка: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
                    dataset_results[method_name] = {
                        "accuracy": metrics['accuracy'], "final_fitness": final_fitness,
                        "n_selected": metrics['n_selected'],
                        "selected_indices_preproc": indices_preproc_final,
                        "selected_indices_original": original_indices_final,
                        "fitness_history": fit_hist, "n_selected_history": n_sel_hist,
                        "extended_metrics": metrics
                    }
                    plot_convergence(fit_hist, n_sel_hist, title=f"Сходимость {method_name} ({dataset_name})")
                except Exception as e:
                    print(f"Ошибка при выполнении метода {method_name}: {e}")
                    dataset_results[method_name] = {
                         "accuracy": 0.0, "final_fitness": -np.inf, "n_selected": -1,
                         "selected_indices_preproc": [], "selected_indices_original": [],
                         "extended_metrics": evaluate_selection_method_extended(X_preprocessed, y, []),
                         "fitness_history": [], "n_selected_history": [], "error": str(e)
                     }
        else: # Если нет признаков после ANOVA
             print("Пропускаем запуск остальных методов FA (нет признаков после ANOVA).")
             for method_name in methods_final_run: # Добавляем пустые результаты
                 dataset_results[method_name] = {
                     "accuracy": 0.0, "final_fitness": 0.0, "n_selected": 0, "selected_indices_preproc": [], "selected_indices_original": [],
                     "extended_metrics": evaluate_selection_method_extended(X_preprocessed, y, []), "fitness_history": [], "n_selected_history": []
                 }

        # 7. Итоги по датасету и визуализация
        print(f"\n--- Сводные результаты для датасета {dataset_name} ---")
        results_summary = []
        for m, res in dataset_results.items(): # Собираем данные для сводной таблицы
            acc = res.get('accuracy', 'N/A'); n_sel_preproc = res.get('n_selected', 'N/A')
            fit = res.get('final_fitness', 'N/A'); roc = res.get('extended_metrics', {}).get('roc_auc', 'N/A')
            f1 = res.get('extended_metrics', {}).get('f1', 'N/A')
            original_indices = res.get('selected_indices_original', [])
            n_sel_orig = len(original_indices) if isinstance(original_indices, list) else 'N/A'
            indices_str = "Нет"
            if original_indices:
                max_indices_to_show = 20
                indices_str = ", ".join(map(str, original_indices[:max_indices_to_show]))
                if len(original_indices) > max_indices_to_show: indices_str += f", ... ({n_sel_orig} всего)"
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            n_sel_orig_str = str(n_sel_orig)
            fit_str = f"{fit:.4f}" if isinstance(fit, float) else str(fit)
            roc_str = f"{roc:.4f}" if isinstance(roc, float) else str(roc)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
            print(f"Метод: {m:<35} | Признаков (ориг): {n_sel_orig_str:<5} | Accuracy: {acc_str:<8} | ROC AUC: {roc_str:<8} | F1: {f1_str:<8}")
            print(f"  -> Fitness: {fit_str:<10} | Признаков (отн. X_prep): {str(n_sel_preproc):<5} | Оригинальные индексы: {indices_str}")
            results_summary.append({'Method': m, 'Features (Orig)': n_sel_orig, 'Features (Prep)': n_sel_preproc,
                                    'Accuracy': acc, 'ROC_AUC': roc, 'F1': f1, 'Fitness': fit,
                                    'Original_Indices': original_indices})
        summary_df = pd.DataFrame(results_summary)
        print("\nСводная таблица (без списка индексов):")
        summary_df_printable = summary_df.drop(columns=['Original_Indices'], errors='ignore')
        print(summary_df_printable.round(4))
        overall_results[dataset_name] = dataset_results
        plot_accuracy_comparison(dataset_results, dataset_name) # Визуализация accuracy

    # --- Конец основного цикла по датасетам ---
    print("\n================== Все датасеты обработаны ==================")
    # Сохранение всех результатов в файл
    import pickle
    with open('firefly_fs_results_final.pkl', 'wb') as f:
        pickle.dump(overall_results, f)
    print("\nПолные результаты сохранены в firefly_fs_results_final.pkl")