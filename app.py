import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, levy_stable, t
from arch import arch_model
import warnings
import scipy.stats as stats

# --- 1. Настройки страницы и стили ---
st.set_page_config(
    page_title="Анализ финансовых рисков",
    page_icon="📈",
    layout="wide"
)
warnings.filterwarnings("ignore")
st.markdown("""
<style>
.st-emotion-cache-16txtl3 {
    margin-top: -75px;
}
.st-emotion-cache-1y4p8pa {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# --- 2. Функции приложения ---

def format_currency(value):
    """Форматирует число в валютный формат (K, M, B)."""
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"


@st.cache_data
def load_data(ticker, start_date, end_date):
    """Загружает исторические данные с Yahoo Finance с максимальной надежностью."""
    try:
        today = pd.to_datetime("today").date()

        # Исправление: правильное сравнение дат
        end_date_converted = pd.to_datetime(end_date).date()
        if end_date_converted > today:
            end_date = today

        try:
            # Добавляем auto_adjust=False для избежания MultiIndex
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )
        except ValueError as ve:
            if "The truth value of a Series is ambiguous" in str(ve):
                st.error(
                    "**Критическая ошибка в библиотеке `yfinance`!**\n\n"
                    "Произошла внутренняя ошибка при загрузке данных. Это известная проблема, которая иногда возникает в `yfinance` из-за формата данных от Yahoo Finance.\n\n"
                    "**Что можно попробовать:**\n"
                    "1. Немного изменить **диапазон дат** (иногда помогает сдвиг на 1-2 дня).\n"
                    "2. Попробовать еще раз через некоторое время.\n"
                    "3. Выбрать другой тикер для анализа."
                )
                return None, None, None
            else:
                st.error(f"Произошла ошибка значения при загрузке данных: {ve}")
                return None, None, None

        # Проверка на пустые данные
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            st.error(
                f"Не удалось загрузить данные для тикера '{ticker}'. Возможно, тикер неверен или нет данных за выбранный период.")
            return None, None, None

        # Обработка MultiIndex колонок (когда загружается несколько тикеров)
        if isinstance(data.columns, pd.MultiIndex):
            # Берем данные для первого тикера
            data = data.xs(ticker, level=1, axis=1, drop_level=True)

        # Поиск колонки с ценой
        price_col = None
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
            price_col = 'Adj Close'
        elif 'Close' in data.columns and not data['Close'].isnull().all():
            price_col = 'Close'
            st.warning(
                "Колонка 'Adj Close' не найдена. Анализ будет выполнен по колонке 'Close', что может быть менее точно (без учета дивидендов).")
        else:
            st.error(f"В загруженных данных для '{ticker}' отсутствуют валидные колонки цен ('Adj Close' или 'Close').")
            return None, None, None

        if len(data) < 2:
            st.error(f"Слишком мало данных для '{ticker}' за выбранный период для расчета доходности.")
            return None, None, None

        # Расчет логарифмической доходности
        log_returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()

        if log_returns.empty:
            st.error(f"Не удалось рассчитать лог-доходность для '{ticker}'. Возможно, в данных много пропусков.")
            return None, None, None

        return data, log_returns, price_col

    except Exception as e:
        st.error(f"Произошла совершенно непредвиденная ошибка: {e}")
        import traceback
        st.error(f"Детали: {traceback.format_exc()}")
        return None, None, None


def fit_levy_stable_fast(returns_data, tail_percentile=0.70):
    """
    Быстрая эмпирическая оценка параметров стабильного распределения.
    Использует метод регрессии хвостов (Tail Index) для оценки Alpha.

    Args:
        returns_data: Данные доходности
        tail_percentile: Порог для определения хвоста (0.70 = топ 30%, 0.90 = топ 10%)
    """
    # Преобразуем в массив numpy и убираем NaN
    x = returns_data.values if isinstance(returns_data, pd.Series) else returns_data
    x = x[~np.isnan(x)]

    if len(x) < 10:
        return 1.8, 0.0, np.median(x), np.std(x)

    # 1. Оценка Loc (Медиана) и Scale (через IQR)
    # Для нормального распр. IQR = 1.349 * sigma.
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    loc = q50
    scale = (q75 - q25) / 1.349

    # 2. Оценка Alpha (Эмпирический метод: регрессия хвостов Log-Log)
    # Alpha определяет наклон хвоста распределения в логарифмическом масштабе.
    try:
        abs_dev = np.abs(x - loc)
        sorted_dev = np.sort(abs_dev)
        n = len(sorted_dev)

        # Используем порог из аргументов
        cutoff_idx = int(n * tail_percentile)

        if cutoff_idx < n - 5:
            tail_data = sorted_dev[cutoff_idx:]

            # Эмпирическая функция выживания: P(X > x)
            # ln(rank) ~ C - alpha * ln(x)
            log_x = np.log(tail_data)
            ranks = np.arange(len(tail_data), 0, -1)
            log_y = np.log(ranks)

            # Линейная регрессия
            slope, _ = np.polyfit(log_x, log_y, 1)
            alpha = -slope
        else:
            alpha = 1.8
    except:
        alpha = 1.8

    # Ограничиваем alpha разумными для финансов пределами [1.0, 2.0]
    # Рынки редко имеют alpha < 1.0 (бесконечное мат. ожидание) или > 2.0
    alpha = max(1.0, min(2.0, alpha))

    # 3. Оценка Beta (Асимметрия)
    # Используем простую квантильную оценку асимметрии
    try:
        q05, q95 = np.percentile(x, [5, 95])
        numer = (q95 - loc) - (loc - q05)
        denom = (q95 - q05)
        beta = numer / denom if denom > 0 else 0.0
    except:
        beta = 0.0

    # Ограничиваем Beta
    beta = max(-1.0, min(1.0, beta))

    return alpha, beta, loc, scale


def plot_distributions_pdf(log_returns, fit_params, ticker):
    """Строит график сравнения PDF."""
    g_mu, g_std = fit_params['gaussian']
    ls_alpha, ls_beta, ls_loc, ls_scale = fit_params['levy']
    garch_fit = fit_params['garch']
    garch_params = garch_fit.params
    nu = garch_params['nu']
    last_vol = garch_fit.conditional_volatility.iloc[-1] / 100

    st.subheader("2. Оцененные параметры моделей")

    # Маркер того, что используется "худшая" alpha
    alpha_label = f"{ls_alpha:.4f}"
    if ls_alpha < 1.7:
        alpha_label += " (Stress Test)"

    param_data = {
        "Параметр": ["Среднее (μ) / Локация (loc)", "Ст. откл. (σ) / Масштаб (scale)", "Индекс стабильности (α)",
                     "Асимметрия (β)", "Ст. волатильность (ω)", "ARCH (α[1])", "GARCH (β[1])", "Форма (ν)"],
        "Гауссова": [f"{g_mu:.5f}", f"{g_std:.5f}", "-", "-", "-", "-", "-", "-"],
        "Леви-стабильная": [f"{ls_loc:.5f}", f"{ls_scale:.5f}", alpha_label, f"{ls_beta:.4f}", "-", "-", "-",
                            "-"],
        "GARCH(1,1)-t": [f"{garch_params['mu'] / 100:.5f}", f"{last_vol:.5f} (условное)", "-", "-",
                         f"{garch_params['omega']:.5f}", f"{garch_params['alpha[1]']:.4f}",
                         f"{garch_params['beta[1]']:.4f}", f"{nu:.4f}"]
    }
    st.dataframe(pd.DataFrame(param_data).set_index("Параметр"), use_container_width=True)

    # Интерпретация параметров
    st.subheader("📖 Интерпретация ключевых параметров")

    interp_col1, interp_col2 = st.columns(2)
    with interp_col1:
        if ls_alpha < 1.5:
            alpha_status = "🔴 **Критически низкий**"
            alpha_text = "Экстремально высокий риск катастрофических событий!"
        elif ls_alpha < 1.8:
            alpha_status = "🟠 **Низкий**"
            alpha_text = "Значительная вероятность 'черных лебедей'"
        elif ls_alpha < 2.0:
            alpha_status = "🟡 **Средний**"
            alpha_text = "Умеренные 'толстые хвосты'"
        else:
            alpha_status = "🟢 **Нормальный**"
            alpha_text = "Близко к нормальному распределению"

        st.markdown(f"**Индекс стабильности α = {ls_alpha:.3f}**")
        st.markdown(f"{alpha_status}")
        st.write(alpha_text)

    with interp_col2:
        if nu < 5:
            nu_status = "🔴 **Очень низкая**"
            nu_text = "Экстремально тяжелые хвосты"
        elif nu < 10:
            nu_status = "🟠 **Низкая**"
            nu_text = "Тяжелые хвосты, высокий риск"
        elif nu < 20:
            nu_status = "🟡 **Средняя**"
            nu_text = "Умеренно тяжелые хвосты"
        else:
            nu_status = "🟢 **Высокая**"
            nu_text = "Близко к нормальному"

        st.markdown(f"**Форма t-распределения ν = {nu:.2f}**")
        st.markdown(f"{nu_status}")
        st.write(nu_text)

    st.subheader("3. Визуальное сравнение плотностей")

    # --- УПРАВЛЕНИЕ ГРАФИКОМ ---
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        use_log_scale = st.checkbox("🔍 Логарифмическая шкала", value=True,
                                    help="Включите, чтобы увидеть разницу в хвостах распределений")

    with col_ctrl2:
        zoom_tails = st.checkbox("🔭 Увеличить хвосты", value=False,
                                 help="Показать отдельные графики для левого и правого хвоста распределения")

    tail_threshold = 0.05
    if zoom_tails:
        with col_ctrl3:
            tail_pct = st.slider("Показать крайние % (хвосты)", 1, 20, 5, 1,
                                 help="Сколько процентов самых экстремальных значений показать")
            tail_threshold = tail_pct / 100.0

    # Общая ось X для отрисовки линий
    # Расширяем диапазон, чтобы хвосты не обрезались
    x_min, x_max = log_returns.min(), log_returns.max()
    margin = (x_max - x_min) * 0.2
    x_full = np.linspace(x_min - margin, x_max + margin, 2000)

    # Предварительный расчет теоретических PDF (чтобы не дублировать код)
    pdf_norm = norm.pdf(x_full, g_mu, g_std)
    pdf_levy = levy_stable.pdf(x_full, ls_alpha, ls_beta, ls_loc, ls_scale)
    pdf_garch = t.pdf(x_full, df=nu, loc=garch_params['mu'] / 100, scale=last_vol)

    if not zoom_tails:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Гистограмма эмпирических данных
        ax.hist(log_returns, bins=150, density=True, alpha=0.5, label=f'Эмпирические данные ({ticker})',
                color='lightblue',
                edgecolor='blue')

        # Теоретические распределения
        ax.plot(x_full, pdf_norm, 'r-', lw=3, label=f'Гауссово (Normal)')
        ax.plot(x_full, pdf_levy, 'g-', lw=3, label=f'Леви-стабильное (Stress α={ls_alpha:.2f})')
        ax.plot(x_full, pdf_garch, 'm-', lw=3, label=f'GARCH-t (Current)')

        # --- ПРИМЕНЕНИЕ ЛОГАРИФМИЧЕСКОГО МАСШТАБА ---
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.001)  # Обрезаем слишком низкие значения, чтобы график был чище
            scale_title = " (Логарифмическая шкала)"
            st.caption(
                "ℹ️ В логарифмической шкале обратите внимание, как зеленая и фиолетовая линии проходят **выше** красной на краях графика.")
        else:
            scale_title = ""
            ax.set_ylim(top=ax.get_ylim()[1] * 1.05)

        ax.set_title(f'Сравнение плотностей распределений дневной доходности {ticker}{scale_title}', fontsize=14,
                     fontweight='bold')
        ax.set_xlabel('Логарифмическая доходность', fontsize=12)
        ax.set_ylabel('Плотность вероятности', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4, which='both')  # which='both' включает сетку для лог шкалы

        # Немного расширяем границы по X, чтобы было видно хвосты
        ax.set_xlim(log_returns.min() * 1.1, log_returns.max() * 1.1)

        plt.tight_layout()
        st.pyplot(fig)

    else:
        # --- РЕЖИМ ЗУМА: ДВА ГРАФИКА ДЛЯ ХВОСТОВ ---
        q_left = log_returns.quantile(tail_threshold)
        q_right = log_returns.quantile(1 - tail_threshold)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # --- ЛЕВЫЙ ХВОСТ (Падения) ---
        ax1.hist(log_returns, bins=300, density=True, alpha=0.5, color='lightblue', edgecolor='blue')
        ax1.plot(x_full, pdf_norm, 'r-', lw=3, label='Гауссово')
        ax1.plot(x_full, pdf_levy, 'g-', lw=3, label='Леви-стабильное')
        ax1.plot(x_full, pdf_garch, 'm-', lw=3, label='GARCH-t')

        ax1.set_xlim(log_returns.min() * 1.1, q_left)  # Зум влево
        ax1.set_title(f"📉 Левый хвост (Худшие {tail_pct}%)", fontsize=12, fontweight='bold')
        ax1.set_ylabel('Плотность', fontsize=10)
        ax1.set_xlabel('Доходность', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.4, which='both')

        # --- ПРАВЫЙ ХВОСТ (Рост) ---
        ax2.hist(log_returns, bins=300, density=True, alpha=0.5, color='lightblue', edgecolor='blue')
        ax2.plot(x_full, pdf_norm, 'r-', lw=3, label='Гауссово')
        ax2.plot(x_full, pdf_levy, 'g-', lw=3, label='Леви-стабильное')
        ax2.plot(x_full, pdf_garch, 'm-', lw=3, label='GARCH-t')

        ax2.set_xlim(q_right, log_returns.max() * 1.1)  # Зум вправо
        ax2.set_title(f"📈 Правый хвост (Лучшие {tail_pct}%)", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Доходность', fontsize=10)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.4, which='both')

        if use_log_scale:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            y_min_zoom = 0.0001
            ax1.set_ylim(bottom=y_min_zoom)
            ax2.set_ylim(bottom=y_min_zoom)
        else:
            # Авто-масштаб Y
            mask_left = x_full <= q_left
            mask_right = x_full >= q_right

            max_y_left = max(
                pdf_norm[mask_left].max() if np.any(mask_left) else 1,
                pdf_levy[mask_left].max() if np.any(mask_left) else 1,
                pdf_garch[mask_left].max() if np.any(mask_left) else 1
            )
            max_y_right = max(
                pdf_norm[mask_right].max() if np.any(mask_right) else 1,
                pdf_levy[mask_right].max() if np.any(mask_right) else 1,
                pdf_garch[mask_right].max() if np.any(mask_right) else 1
            )

            ax1.set_ylim(0, max_y_left * 1.5)
            ax2.set_ylim(0, max_y_right * 1.5)

        plt.tight_layout()
        st.pyplot(fig)

    st.info(
        "💡 **Ключевой вывод:** Обратите внимание на **хвосты распределения**. "
        "В данной версии график 'Леви' построен на основе **худшего** значения Alpha из истории (Stress Test), "
        "чтобы показать максимальный потенциальный риск."
    )


def run_and_plot_var_simulation(fit_params, capital, horizon, confidence, sims=10000):
    """Проводит симуляцию Монте-Карло и оценивает VaR."""
    st.subheader("1. Результаты симуляции Value-at-Risk (VaR)")
    with st.spinner(f"Запуск симуляции Монте-Карло ({sims} сценариев)..."):
        g_mu, g_std = fit_params['gaussian']
        g_returns_sim = norm.rvs(loc=g_mu, scale=g_std, size=(horizon, sims))
        final_capital_g = capital * np.exp(g_returns_sim.sum(axis=0))
        losses_g = capital - final_capital_g
        var_g = np.percentile(losses_g, confidence)

        ls_alpha, ls_beta, ls_loc, ls_scale = fit_params['levy']
        ls_returns_sim = levy_stable.rvs(alpha=ls_alpha, beta=ls_beta, loc=ls_loc, scale=ls_scale, size=(horizon, sims))
        final_capital_ls = capital * np.exp(ls_returns_sim.sum(axis=0))
        losses_ls = capital - final_capital_ls
        var_ls = np.percentile(losses_ls, confidence)

        garch_fit = fit_params['garch']
        forecasts = garch_fit.forecast(horizon=horizon, method='simulation', simulations=sims)
        sim_returns_garch_pct = forecasts.simulations.values[0].T
        sim_returns_garch = np.log(sim_returns_garch_pct / 100 + 1)
        final_capital_garch = capital * np.exp(sim_returns_garch.sum(axis=0))
        losses_garch = capital - final_capital_garch
        var_garch = np.percentile(losses_garch, confidence)

    st.write(f"**{confidence}% VaR на горизонте {horizon} дней для портфеля в ${capital:,.0f}**")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="VaR (Гаусс)", value=f"${var_g:,.0f}",
                help=f"С вероятностью {100 - confidence:.1f}% убыток НЕ превысит эту сумму.")
    col2.metric(label="VaR (Леви-стабильная)", value=f"${var_ls:,.0f}", delta=f"{((var_ls - var_g) / var_g):.1%}",
                delta_color="inverse",
                help="Дельта показывает разницу с Гауссовой моделью. Используется Worst Case Alpha.")
    col3.metric(label="VaR (GARCH-t)", value=f"${var_garch:,.0f}", delta=f"{((var_garch - var_g) / var_g):.1%}",
                delta_color="inverse", help="Дельта показывает разницу с Гауссовой моделью.")

    st.subheader("2. Визуализация распределения смоделированных убытков")

    # Диагностическая информация
    with st.expander("📊 Статистика симуляций (для отладки)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Гауссова модель:**")
            st.write(f"Min убыток: ${losses_g.min():,.0f}")
            st.write(f"Max убыток: ${losses_g.max():,.0f}")
            st.write(f"Median: ${np.median(losses_g):,.0f}")
        with col_b:
            st.write("**Леви-стабильная:**")
            st.write(f"Min убыток: ${losses_ls.min():,.0f}")
            st.write(f"Max убыток: ${losses_ls.max():,.0f}")
            st.write(f"Median: ${np.median(losses_ls):,.0f}")
        with col_c:
            st.write("**GARCH-t:**")
            st.write(f"Min убыток: ${losses_garch.min():,.0f}")
            st.write(f"Max убыток: ${losses_garch.max():,.0f}")
            st.write(f"Median: ${np.median(losses_garch):,.0f}")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Улучшенный расчет диапазона для графика
    all_losses = np.concatenate([losses_g, losses_ls, losses_garch])
    # Используем перцентили для определения разумного диапазона (отсекаем экстремальные выбросы)
    lower_bound = np.percentile(all_losses, 0.1)
    upper_bound = np.percentile(all_losses, 99.9)

    # Создаем бины для гистограммы
    bins = np.linspace(lower_bound, upper_bound, 100)

    ax.hist(losses_g, bins=bins, density=True, alpha=0.6, label='Гауссова модель', color='red', edgecolor='darkred')
    ax.hist(losses_ls, bins=bins, density=True, alpha=0.6, label='Леви-стабильная (Stress)', color='green',
            edgecolor='darkgreen')
    ax.hist(losses_garch, bins=bins, density=True, alpha=0.6, label='GARCH-t модель', color='purple',
            edgecolor='indigo')

    # Добавляем линии VaR
    ax.axvline(var_g, color='darkred', linestyle='--', lw=2.5,
               label=f'{confidence}% VaR (Гаусс): {format_currency(var_g)}')
    ax.axvline(var_ls, color='darkgreen', linestyle='--', lw=2.5,
               label=f'{confidence}% VaR (Леви): {format_currency(var_ls)}')
    ax.axvline(var_garch, color='indigo', linestyle='--', lw=2.5,
               label=f'{confidence}% VaR (GARCH-t): {format_currency(var_garch)}')

    ax.set_title(f'Распределение убытков портфеля за {horizon} дней ({sims} симуляций)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Убыток ($)', fontsize=12)
    ax.set_ylabel('Плотность вероятности', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Форматирование оси X
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x)))

    st.pyplot(fig)

    # Добавляем интерпретацию результатов
    st.info(
        f"**💡 Интерпретация результатов:**\n\n"
        f"• **Гауссова модель** предполагает 'нормальное' распределение и может **недооценивать** экстремальные риски.\n"
        f"• **Леви-стабильная** здесь работает в режиме **стресс-теста** (использует худшее значение Alpha за историю). VaR на {abs((var_ls - var_g) / var_g * 100):.1f}% {'выше' if var_ls > var_g else 'ниже'}.\n"
        f"• **GARCH-t** учитывает изменяющуюся во времени волатильность.\n\n"
        f"Чем больше разница между моделями, тем важнее учитывать 'хвостовые риски' при принятии решений."
    )

    # Возвращаем VaR для использования в финальном резюме
    return var_g, var_ls, var_garch


@st.cache_data
def calculate_rolling_alpha(log_returns, window_size, tail_cutoff=0.70):
    """Рассчитывает параметр alpha в скользящем окне."""
    if len(log_returns) < window_size:
        return None

    def safe_levy_fit(x):
        if len(x) >= 100:
            try:
                # Быстрая подгонка с ограничением выборки
                alpha, _, _, _ = fit_levy_stable_fast(x, tail_percentile=tail_cutoff)
                return alpha
            except:
                return np.nan
        return np.nan

    rolling_alpha = log_returns.rolling(window=window_size).apply(safe_levy_fit, raw=False)
    return rolling_alpha.dropna()


def plot_rolling_alpha(log_returns, window_size, ticker, tail_cutoff):
    """Отображает график rolling alpha."""
    st.subheader("Динамика параметра стабильности α (Alpha)")
    st.write("""
        Параметр **α (alpha)** Леви-стабильного распределения определяет "тяжесть хвостов" — то есть, вероятность экстремальных событий.
        - **α = 2:** Нормальное распределение (нет "толстых хвостов").
        - **α < 2:** Распределение с "толстыми хвостами".
        **Чем ниже α, тем выше вероятность "черных лебедей" (катастрофических событий).** Этот график показывает, как менялась оценка этого параметра во времени в скользящем окне.
    """, unsafe_allow_html=True)

    # Проверка достаточности данных
    min_required = window_size + 100  # Нужен запас для расчета
    if len(log_returns) < min_required:
        st.warning(
            f"⚠️ **Недостаточно данных для Rolling Alpha анализа!**\n\n"
            f"• Имеется данных: **{len(log_returns)}** дней\n"
            f"• Требуется минимум: **{min_required}** дней (окно {window_size} + 100 для расчета)\n\n"
            f"**Рекомендации:**\n"
            f"1. Увеличьте период анализа (например, с 2010 года)\n"
            f"2. Уменьшите размер окна (попробуйте 100-150 дней)\n"
            f"3. Для качественного анализа нужно минимум 2-3 года данных"
        )
        return

    with st.spinner(f"Расчет rolling alpha с окном {window_size} дней (чувствительность {tail_cutoff * 100:.0f}%)..."):
        rolling_alpha = calculate_rolling_alpha(log_returns, window_size, tail_cutoff)

    if rolling_alpha is None or rolling_alpha.empty or len(rolling_alpha) < 10:
        st.error(
            "❌ **Не удалось рассчитать rolling alpha.**\n\n"
            "Возможные причины:\n"
            "• Слишком короткий период данных\n"
            "• Размер окна слишком большой относительно периода\n"
            "• Недостаточно вариации в данных для оценки параметра α"
        )
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(rolling_alpha.index, rolling_alpha, label=f'Rolling α (окно {window_size} дней)',
            color='cyan', linewidth=2, alpha=0.8)

    mean_alpha = rolling_alpha.mean()
    min_alpha = rolling_alpha.min()

    ax.axhline(mean_alpha, color='red', linestyle='--', lw=2, label=f'Среднее α = {mean_alpha:.2f}')
    ax.axhline(min_alpha, color='green', linestyle='--', lw=2, label=f'Min α = {min_alpha:.2f} (Stress)')
    ax.axhline(2.0, color='gray', linestyle=':', lw=1.5, label='α = 2 (Нормальное распределение)', alpha=0.7)

    # Добавляем зоны риска
    ax.axhspan(0, 1.5, alpha=0.1, color='red', label='Зона высокого риска (α < 1.5)')
    ax.axhspan(1.5, 1.8, alpha=0.1, color='orange')
    ax.axhspan(1.8, 2.0, alpha=0.1, color='yellow')

    ax.set_title(f'Динамика параметра стабильности α для {ticker}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Значение α (Alpha)', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=max(0, rolling_alpha.min() - 0.1), top=min(2.1, rolling_alpha.max() + 0.1))

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Статистика
    st.subheader("📊 Статистика параметра α")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Среднее α", f"{mean_alpha:.3f}")
    col2.metric("Мин α (Стресс)", f"{rolling_alpha.min():.3f}")
    col3.metric("Макс α", f"{rolling_alpha.max():.3f}")
    col4.metric("Ст. откл.", f"{rolling_alpha.std():.3f}")

    # Интерпретация
    if mean_alpha < 1.5:
        risk_level = "🔴 **ОЧЕНЬ ВЫСОКИЙ**"
        interpretation = "Рынок демонстрирует экстремально высокую вероятность катастрофических событий!"
    elif mean_alpha < 1.8:
        risk_level = "🟠 **ВЫСОКИЙ**"
        interpretation = "Присутствуют значительные 'толстые хвосты' — риск черных лебедей повышен."
    elif mean_alpha < 2.0:
        risk_level = "🟡 **СРЕДНИЙ**"
        interpretation = "Умеренные отклонения от нормального распределения."
    else:
        risk_level = "🟢 **НИЗКИЙ**"
        interpretation = "Распределение близко к нормальному."

    st.info(
        f"**Уровень риска:** {risk_level}\n\n"
        f"{interpretation}\n\n"
        f"**Контекст:** Чем ниже α, тем чаще случаются экстремальные движения цен, "
        f"которые не предсказываются стандартными моделями."
    )

    return rolling_alpha  # Return to be used elsewhere


def plot_qq_charts(log_returns, fit_params):
    """Строит Q-Q графики для всех моделей."""
    st.subheader("Сравнение на квантиль-квантильных (Q-Q) графиках")
    st.write(
        "**Q-Q график** сравнивает квантили эмпирического распределения с теоретическими. "
        "Если точки лежат на красной линии, модель хорошо описывает данные. "
        "Отклонения на концах (хвостах) указывают на недооценку экстремальных событий."
    )

    g_mu, g_std = fit_params['gaussian']
    ls_alpha, ls_beta, ls_loc, ls_scale = fit_params['levy']
    garch_fit = fit_params['garch']

    # Стандартизированные остатки для GARCH
    std_resid = garch_fit.resid / garch_fit.conditional_volatility

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Q-Q против нормального
    stats.probplot(log_returns, dist="norm", sparams=(g_mu, g_std), plot=axes[0])
    axes[0].set_title('Q-Q: Нормальное распределение', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Теоретические квантили', fontsize=10)
    axes[0].set_ylabel('Эмпирические квантили', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Q-Q против Леви-стабильного
    stats.probplot(log_returns, dist=levy_stable, sparams=(ls_alpha, ls_beta, ls_loc, ls_scale), plot=axes[1])
    axes[1].set_title(f'Q-Q: Леви-стабильное (Stress α={ls_alpha:.2f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Теоретические квантили', fontsize=10)
    axes[1].set_ylabel('Эмпирические квантили', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Q-Q остатков GARCH против t-распределения
    stats.probplot(std_resid, dist="t", sparams=(garch_fit.params['nu'],), plot=axes[2])
    axes[2].set_title(f'Q-Q: GARCH-t остатки (ν={garch_fit.params["nu"]:.2f})', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Теоретические квантили', fontsize=10)
    axes[2].set_ylabel('Стандартизированные остатки', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Интерпретация графиков
    st.subheader("📖 Как интерпретировать Q-Q графики")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔴 Нормальное распределение**")
        st.write(
            "Если видны отклонения на концах (особенно на левом нижнем углу), "
            "это означает, что Гауссова модель **недооценивает риск** экстремальных падений рынка."
        )

    with col2:
        st.markdown("**🟢 Леви-стабильное**")
        st.write(
            "Точки должны лучше соответствовать линии на хвостах. "
            "Леви-модель специально разработана для учета 'толстых хвостов' — "
            "редких, но катастрофических событий."
        )

    with col3:
        st.markdown("**🟣 GARCH-t остатки**")
        st.write(
            "GARCH моделирует волатильность, которая меняется во времени. "
            "Хорошее соответствие означает, что модель успешно 'очистила' данные "
            "от кластеров волатильности."
        )

    # Анализ качества подгонки
    st.info(
        "💡 **Совет:** Лучшая модель — та, у которой точки максимально близки к красной линии "
        "**особенно на концах графика** (экстремальные значения). Центральная часть обычно хорошо "
        "описывается всеми моделями."
    )


# --- 3. Пользовательский интерфейс (Боковая панель) ---
st.sidebar.header("⚙️ Параметры анализа")
ticker = st.sidebar.text_input("Тикер актива", value="^GSPC",
                               help="Например: ^GSPC (S&P500), AAPL (Apple), BTC-USD (Bitcoin)")
start_date = st.sidebar.date_input("Дата начала", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Дата окончания", pd.to_datetime("today"))

# Проверка корректности дат
if start_date >= end_date:
    st.sidebar.error("⚠️ Дата начала должна быть раньше даты окончания!")

# Расчет периода и предупреждения
days_diff = (end_date - start_date).days
if days_diff < 365:
    st.sidebar.warning(
        f"⚠️ Выбран короткий период ({days_diff} дней). Рекомендуется минимум 1 год для точного анализа.")

st.sidebar.header("💰 Параметры симуляции VaR")
initial_capital = st.sidebar.number_input(
    "Начальный капитал ($)",
    min_value=1000,
    value=1_000_000,
    step=1000,
    help="Размер портфеля для расчета Value-at-Risk"
)
confidence_level = st.sidebar.slider(
    "Уровень доверия (%)",
    min_value=90.0,
    max_value=99.9,
    value=99.0,
    step=0.5,
    help="Вероятность, что убыток не превысит VaR. Стандартные значения: 95%, 99%, 99.9%"
)
horizon_days = st.sidebar.slider(
    "Горизонт симуляции (дней)",
    min_value=5,
    max_value=252,
    value=30,
    step=1,
    help="Период прогноза. 21 день ≈ 1 месяц, 252 дня ≈ 1 год"
)

st.sidebar.header("📊 Динамический анализ")
rolling_window = st.sidebar.slider(
    "Окно для Rolling Alpha (дней)",
    min_value=100,
    max_value=1000,
    value=252,
    step=50,
    help="Размер скользящего окна. Рекомендуется 252 дня (1 торговый год)"
)

# НОВАЯ НАСТРОЙКА: Чувствительность хвостов для Rolling Alpha
tail_cutoff_percent = st.sidebar.slider(
    "Чувствительность к хвостам (%)",
    min_value=50,
    max_value=99,
    value=90,
    step=1,
    help="Какой % самых сильных движений считать 'хвостом'. 90% = берем только топ-10% кризисов. Чем выше %, тем чувствительнее Alpha."
)
tail_cutoff = tail_cutoff_percent / 100.0

if rolling_window > days_diff - 100:
    st.sidebar.warning(f"⚠️ Окно ({rolling_window} дней) слишком велико для выбранного периода ({days_diff} дней).")

# --- 4. Основная часть приложения ---

st.title(f"📈 Анализ экстремальных рисков для {ticker}")
st.caption(f"Период: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')} ({days_diff} дней)")

# Информационные блоки
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(
        "**🎯 Цель анализа:**\n\n"
        "Сравнить **три подхода** к оценке финансового риска:\n"
        "1. **Гауссова модель** (классическая, но недооценивает риски)\n"
        "2. **Леви-стабильная** (учитывает 'черных лебедей')\n"
        "3. **GARCH-t** (учитывает меняющуюся волатильность)"
    )
with col_info2:
    st.warning(
        "**⚠️ Важно:**\n\n"
        "• Расчет **Rolling Alpha** требует минимум **2-3 года** данных\n"
        "• Для горизонта **252 дня** расчет займет **1-2 минуты**\n"
        "• Большие окна (>500 дней) значительно замедляют расчет"
    )
# Инициализация session_state для хранения результатов
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if st.button("🚀 Запустить анализ", type="primary", use_container_width=True, help="Начать полный анализ рисков"):

    data, log_returns, price_col = load_data(ticker, start_date, end_date)

    if data is not None and log_returns is not None:
        # Прогресс-бар для лучшего UX
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Шаг 1/3: Подгонка Гауссовой модели...")
        progress_bar.progress(10)

        g_mu, g_std = norm.fit(log_returns)
        progress_bar.progress(30)

        status_text.text("Шаг 2/3: Расчет скользящей Alpha и поиск худшего сценария...")

        # 1. Сначала считаем Rolling Alpha, чтобы найти худший сценарий
        rolling_alpha = calculate_rolling_alpha(log_returns, rolling_window, tail_cutoff)

        # Определяем "Stress" Alpha (худшее значение)
        if rolling_alpha is not None and not rolling_alpha.empty:
            worst_case_alpha = rolling_alpha.min()
        else:
            # Fallback
            worst_case_alpha, _, _, _ = fit_levy_stable_fast(log_returns, tail_cutoff)

        # Получаем остальные параметры Леви (можно использовать глобальные или пересчитать)
        # Для простоты используем глобальную подгонку для beta, loc, scale
        _, ls_beta, ls_loc, ls_scale = fit_levy_stable_fast(log_returns, tail_cutoff)

        progress_bar.progress(60)

        status_text.text("Шаг 3/3: Подгонка GARCH(1,1)-t модели...")

        garch = arch_model(log_returns * 100, vol='Garch', p=1, q=1, dist='t')
        garch_fit = garch.fit(disp='off', show_warning=False)

        progress_bar.progress(100)
        status_text.text("✅ Подгонка моделей завершена!")

        # ВАЖНО: Используем worst_case_alpha для модели Леви
        fit_params = {
            "gaussian": (g_mu, g_std),
            "levy": (worst_case_alpha, ls_beta, ls_loc, ls_scale),  # ПОДМЕНА НА ХУДШИЙ СЛУЧАЙ
            "garch": garch_fit
        }

        # Сохраняем результаты в session_state
        st.session_state.data = data
        st.session_state.log_returns = log_returns
        st.session_state.price_col = price_col
        st.session_state.fit_params = fit_params
        st.session_state.analysis_complete = True
        st.session_state.ticker = ticker
        st.session_state.initial_capital = initial_capital
        st.session_state.horizon_days = horizon_days
        st.session_state.confidence_level = confidence_level
        st.session_state.rolling_window = rolling_window
        st.session_state.tail_cutoff = tail_cutoff
        st.session_state.rolling_alpha_series = rolling_alpha  # Сохраняем серию для графика

        # Очищаем индикаторы прогресса
        progress_bar.empty()
        status_text.empty()

        st.success(f"Анализ завершен! Для модели Леви используется стресс-сценарий: Alpha = {worst_case_alpha:.3f}")

# Отображаем результаты только если анализ был выполнен
if st.session_state.analysis_complete:

    tab1, tab2, tab3, tab4 = st.tabs([
        "Обзор и подгонка распределений",
        "Анализ Value-at-Risk (VaR)",
        "Динамический анализ риска (Rolling Alpha)",
        "Сравнение Q-Q"
    ])

    with tab1:
        st.header("📊 Динамика цены и логарифмической доходности")

        # Статистика данных
        st.subheader("1. Основные статистики")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Торговых дней", len(st.session_state.log_returns))
        stat_col2.metric("Средняя доходность", f"{st.session_state.log_returns.mean() * 100:.4f}%")
        stat_col3.metric("Волатильность (дневная)", f"{st.session_state.log_returns.std() * 100:.2f}%")
        stat_col4.metric("Волатильность (годовая)", f"{st.session_state.log_returns.std() * np.sqrt(252) * 100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Цена закрытия ({st.session_state.price_col})")
            st.line_chart(st.session_state.data[st.session_state.price_col])
        with col2:
            st.subheader("Дневная лог-доходность")
            st.line_chart(st.session_state.log_returns)

        st.header("🔬 Сравнение моделей распределений")
        plot_distributions_pdf(st.session_state.log_returns, st.session_state.fit_params, st.session_state.ticker)

    with tab2:
        st.header("💰 Оценка риска с помощью симуляции Монте-Карло")
        var_g, var_ls, var_garch = run_and_plot_var_simulation(
            st.session_state.fit_params,
            st.session_state.initial_capital,
            st.session_state.horizon_days,
            st.session_state.confidence_level
        )

    with tab3:
        st.header("⏰ Анализ стабильности риска во времени")
        # Используем сохраненные значения
        cutoff_val = st.session_state.get('tail_cutoff', 0.70)
        # Используем уже посчитанную серию
        if 'rolling_alpha_series' in st.session_state:
            plot_rolling_alpha(st.session_state.rolling_alpha_series, st.session_state.rolling_window,
                               st.session_state.ticker, cutoff_val)
        else:
            plot_rolling_alpha(
                calculate_rolling_alpha(st.session_state.log_returns, st.session_state.rolling_window, cutoff_val),
                st.session_state.rolling_window, st.session_state.ticker, cutoff_val)

    with tab4:
        st.header("📐 Анализ квантиль-квантиль (Q-Q)")
        plot_qq_charts(st.session_state.log_returns, st.session_state.fit_params)

    # Финальные рекомендации
    st.markdown("---")
    st.header("🎯 Общие выводы и рекомендации")

    conclusion_col1, conclusion_col2 = st.columns(2)

    with conclusion_col1:
        ls_alpha = st.session_state.fit_params['levy'][0]
        st.success(
            "**✅ Что мы узнали:**\n\n"
            f"• Индекс стабильности α = **{ls_alpha:.2f}** "
            f"{'(высокий риск черных лебедей)' if ls_alpha < 1.8 else '(умеренный риск)'}\n\n"
            "• Гауссова модель может **существенно недооценивать** реальные риски\n\n"
            "• GARCH и Леви-модели дают более реалистичную картину"
        )

    with conclusion_col2:
        st.warning(
            "**⚠️ Рекомендации:**\n\n"
            "1. **Не полагайтесь только на Гауссову модель** при оценке рисков\n\n"
            "2. Учитывайте **'толстые хвосты'** — экстремальные события случаются чаще, чем предсказывает нормальное распределение\n\n"
            "3. Используйте **консервативные оценки** VaR (Леви-модель)\n\n"
            "4. Регулярно **пересматривайте** риск-модели на актуальных данных"
        )

else:
    st.info("👆 Настройте параметры на боковой панели и нажмите кнопку **'Запустить анализ'** для начала работы.")
st.sidebar.markdown("---")
if st.session_state.get('analysis_complete', False):
    if st.sidebar.button("🔄 Сбросить анализ", type="secondary", use_container_width=True):
        st.session_state.analysis_complete = False
        st.rerun()
st.sidebar.info(
    "**📚 Инструкция:**\n\n"
    "1️⃣ Введите **тикер** актива ([Yahoo Finance](https://finance.yahoo.com/lookup/))\n\n"
    "2️⃣ Выберите **период** анализа (рекомендуется 3-5 лет)\n\n"
    "3️⃣ Настройте параметры **VaR симуляции**\n\n"
    "4️⃣ Нажмите **'Запустить анализ'**\n\n"
    "5️⃣ Изучите результаты на **4 вкладках**\n\n"
    "---\n\n"
    "**💡 Популярные тикеры:**\n"
    "• `^GSPC` — S&P 500\n"
    "• `^DJI` — Dow Jones\n"
    "• `AAPL` — Apple\n"
    "• `TSLA` — Tesla\n"
    "• `BTC-USD` — Bitcoin\n"
    "• `GC=F` — Золото"
)