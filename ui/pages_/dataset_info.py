from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


def add_percentage_column(df_pivot, group_col, count_col):
    """
    Añade una columna de porcentajes a un DataFrame pivotado.

    Calcula el porcentaje que representa cada valor respecto al total de su grupo.

    Parameters:
        df_pivot (pd.DataFrame): DataFrame con los datos agrupados
        group_col (str): Nombre de la columna que define los grupos
        count_col (str): Nombre de la columna con los valores a convertir en porcentajes

    Returns:
        pd.DataFrame: DataFrame original con la nueva columna 'pct' añadida
    """
    total_per_group = df_pivot.groupby(group_col)[count_col].transform("sum")
    df_pivot["pct"] = df_pivot[count_col] / total_per_group
    return df_pivot


def maybe_format_y(fig, view_mode, y_col):
    """
        Configura el formato del eje Y y las etiquetas de texto en un gráfico de Plotly.

        Ajusta el formato según el modo de visualización seleccionado (cantidad o porcentaje).

        Parameters:
            fig (plotly.graph_objects.Figure): Figura de Plotly a modificar
            view_mode (str): Modo de visualización ('Porcentaje' o 'Cantidad')
            y_col (str): Nombre de la columna utilizada para el eje Y

        Returns:
            None: La figura se modifica in-place
        """
    if view_mode == "Porcentaje":
        fig.update_yaxes(tickformat=".0%")
    else:
        fig.update_yaxes(tickformat="d")
    fig.update_traces(
        texttemplate="%{text:.1%}" if view_mode == "Porcentaje" else "%{text}"
    )


def dataset_info_page() -> None:
    """
    Genera el dashboard de información sobre reservas de hotel.

    Crea una interfaz interactiva con Streamlit que muestra:
    - Filtros para tipo de hotel, año y mes de llegada
    - Métricas principales (reservas, cancelaciones, tasa, ADR)
    - Visualizaciones en pestañas para analizar diferentes aspectos:
      * Cancelaciones por mes
      * Días de antelación
      * Precio medio
      * Países con más reservas
      * Segmentos de mercado

    Returns:
        None: Renderiza directamente en la interfaz de Streamlit
    """
    st.title("🏨 Hotel Booking Dashboard")
    st.caption("Explora tus reservas de forma visual y sencilla")

    @st.cache_data
    def load_data(path: Path | str) -> pd.DataFrame:
        return pd.read_csv(path)

    DATA_FILE = "data/dataset_practica_final.csv"
    df = load_data(DATA_FILE)

    with st.container():
        c1, c2, c3 = st.columns(3)

        hotel_sel = c1.multiselect("Tipo de hotel",
                                   options=df["hotel"].unique(),
                                   default=df["hotel"].unique())

        year_sel = c2.multiselect("Año de llegada",
                                  options=sorted(df["arrival_date_year"].unique()),
                                  default=sorted(df["arrival_date_year"].unique()))

        month_sel = c3.multiselect("Mes de llegada",
                                   options=list(df["arrival_date_month"].unique()),
                                   default=list(df["arrival_date_month"].unique()))

    mask = (
        df["hotel"].isin(hotel_sel)
        & df["arrival_date_year"].isin(year_sel)
        & df["arrival_date_month"].isin(month_sel)
    )
    data = df[mask]

    view_mode = st.radio("Modo de visualización",
                         ["Cantidad", "Porcentaje"],
                         horizontal=True)

    total_res = len(data)
    total_cancel = int(data["is_canceled"].sum())
    cancel_rate = total_cancel / total_res if total_res else 0
    mean_adr = data["adr"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reservas", f"{total_res:,}")
    m2.metric("Cancelaciones", f"{total_cancel:,}")
    m3.metric("Tasa cancelación", f"{cancel_rate:.1%}")
    m4.metric("ADR medio (€)", f"{mean_adr:,.2f}")

    st.divider()

    tabs = st.tabs([
        "📅 Cancelaciones por mes",
        "⏳ Antelación",
        "💰 Precio medio",
        "🌍 Países top",
        "📊 Segmento de mercado"
    ])

    from calendar import month_name
    month_order = list(month_name)[1:]

    with tabs[0]:
        st.subheader("📅 Cancelaciones por mes")
        pivot = (
            data.groupby(["arrival_date_month", "is_canceled"])
            .size()
            .reset_index(name="count")
            .sort_values("arrival_date_month",
                         key=lambda s: s.map({m: i for i, m in enumerate(month_order)}))
        )
        if view_mode == "Porcentaje":
            pivot = add_percentage_column(pivot, "arrival_date_month", "count")
            y_col, text_col = "pct", "pct"
        else:
            y_col, text_col = "count", "count"

        fig1 = px.bar(
            pivot,
            x="arrival_date_month",
            y=y_col,
            color="is_canceled",
            category_orders={"arrival_date_month": month_order},
            labels={"is_canceled": "¿Cancelada? (0 = No, 1 = Sí)",
                    "arrival_date_month": "Mes",
                    y_col: "Porcentaje" if view_mode == "Porcentaje" else "Reservas"},
            barmode="stack",
            text=text_col,
            height=430,
        )
        maybe_format_y(fig1, view_mode, y_col)
        st.plotly_chart(fig1, use_container_width=True)

    with tabs[1]:
        st.subheader("⏳ Días de antelación (Lead Time)")
        fig2 = px.histogram(
            data,
            x="lead_time",
            nbins=40,
            color="is_canceled",
            labels={"is_canceled": "¿Cancelada? (0 = No, 1 = Sí)",
                    "lead_time": "Días de antelación"},
            height=430,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.subheader("💰 Precio medio (ADR)")
        mean_prices = (
            data.groupby("is_canceled")["adr"]
            .mean()
            .reset_index()
            .replace({"is_canceled": {0: "Confirmadas", 1: "Canceladas"}})
        )
        fig3 = px.bar(
            mean_prices,
            x="is_canceled",
            y="adr",
            color="is_canceled",
            text_auto=".2f",
            labels={"is_canceled": "", "adr": "ADR medio (€)"},
            height=430,
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tabs[3]:
        st.subheader("🌍 Países con más reservas")
        top_countries = (
            data["country"]
            .value_counts(normalize=(view_mode == "Porcentaje"))
            .head(10)
            .rename_axis("country")
            .reset_index(name="value")
        )
        fig4 = px.bar(
            top_countries,
            x="country",
            y="value",
            text="value",
            labels={
                "value": "% de reservas" if view_mode == "Porcentaje" else "Reservas",
                "country": "País",
            },
            height=430,
        )
        if view_mode == "Porcentaje":
            fig4.update_yaxes(tickformat=".0%")
            fig4.update_traces(texttemplate="%{text:.1%}")
        else:
            fig4.update_traces(texttemplate="%{text:d}")
        st.plotly_chart(fig4, use_container_width=True)

    with tabs[4]:
        st.subheader("📊 Distribución por segmento de mercado")
        seg = (
            data.groupby(["market_segment", "is_canceled"])
            .size()
            .reset_index(name="count")
        )
        if view_mode == "Porcentaje":
            seg = add_percentage_column(seg, "market_segment", "count")
            y_col, text_col = "pct", "pct"
        else:
            y_col, text_col = "count", "count"
        fig5 = px.bar(
            seg,
            x="market_segment",
            y=y_col,
            color="is_canceled",
            labels={
                "is_canceled": "¿Cancelada? (0 = No, 1 = Sí)",
                "market_segment": "Segmento",
                y_col: "% dentro segmento" if view_mode == "Porcentaje" else "Reservas",
            },
            barmode="stack",
            text=text_col,
            height=430,
        )
        maybe_format_y(fig5, view_mode, y_col)
        st.plotly_chart(fig5, use_container_width=True)

    with st.expander("🔍 Ver datos filtrados"):
        st.dataframe(data, use_container_width=True)


