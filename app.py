
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="CrimeSense",
    page_icon="🔎",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "crimesense_final_pipeline.pkl"
)

st.title("🔎 CrimeSense")
st.caption(
    "District-Level Crime Rate Prediction and Hotspot Analysis"
)

# -------------------- LOAD DATA --------------------

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

district_data = load_csv(
    os.path.join(DATA_DIR, "district_year_data.csv")
)

hotspot_data = load_csv(
    os.path.join(DATA_DIR, "district_hotspot_ranking_2023.csv")
)

evaluation = load_csv(
    os.path.join(DATA_DIR, "final_model_evaluation.csv")
)

model = load_model(MODEL_PATH)

# -------------------- SIDEBAR --------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Crime Analysis",
        "Risk Classification",
        "Hotspot Analysis",
        "Model Performance",
        "Crime Prediction"
    ]
)

if district_data.empty:
    st.error("District-year dataset not found. Check the data file path.")
    st.stop()

# -------------------- HOME DASHBOARD --------------------

if page == "Home Dashboard":
    st.header("Project Overview")

    latest_year = int(district_data["Year"].max())
    latest = district_data[
        district_data["Year"] == latest_year
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric("Districts", latest["District"].nunique())
    c2.metric("Year", latest_year)
    c3.metric("Total Recorded Cases", f'{int(latest["Total_Cases"].sum()):,}')

    st.subheader("Annual Crime Rate Trend")

    annual = (
        district_data.groupby("Year", as_index=False)
        ["Crime_Rate_per_100k"].mean()
    )

    fig = px.line(
        annual,
        x="Year",
        y="Crime_Rate_per_100k",
        markers=True,
        title="Average District Crime Rate by Year"
    )
    st.plotly_chart(fig, use_container_width=True)
    # ---------- Year-over-Year Crime Rate Insight ----------
if len(annual) >= 2:
    annual = annual.sort_values("Year")

    previous_rate = float(
        annual.iloc[-2]["Crime_Rate_per_100k"]
    )
    latest_rate = float(
        annual.iloc[-1]["Crime_Rate_per_100k"]
    )

    change = latest_rate - previous_rate

    if previous_rate != 0:
        change_pct = (change / previous_rate) * 100
    else:
        change_pct = None

    st.subheader("Year-over-Year Insight")

    col1, col2 = st.columns(2)

    col1.metric(
        f"Average Crime Rate ({int(annual.iloc[-1]['Year'])})",
        f"{latest_rate:.2f}",
        delta=(
            f"{change_pct:+.2f}% vs previous year"
            if change_pct is not None
            else None
        ),
        delta_color="inverse"
    )

    col2.metric(
        "Change in Crime Rate",
        f"{change:+.2f}",
        help="Difference between the latest two available annual averages."
    )

    if change > 0:
        st.warning(
            "The average recorded crime rate increased compared "
            "with the previous available year."
        )
    elif change < 0:
        st.success(
            "The average recorded crime rate decreased compared "
            "with the previous available year."
        )
    else:
        st.info(
            "The average recorded crime rate remained unchanged."
        )

    st.info(
        "CrimeSense analyzes historical district-level data. "
        "Results depend on the quality and coverage of the dataset."
    )

# -------------------- CRIME ANALYSIS --------------------

elif page == "Crime Analysis":
    st.header("Crime Analysis")

    years = sorted(district_data["Year"].unique())
    selected_year = st.selectbox("Select year", years, index=len(years)-1)

    year_data = district_data[
        district_data["Year"] == selected_year
    ].copy()

    state_summary = (
        year_data.groupby("State", as_index=False)
        .agg(
            Average_Crime_Rate=("Crime_Rate_per_100k", "mean"),
            Total_Cases=("Total_Cases", "sum")
        )
        .sort_values("Average_Crime_Rate", ascending=False)
    )

    fig = px.bar(
        state_summary,
        x="State",
        y="Average_Crime_Rate",
        title=f"Average District Crime Rate by State — {selected_year}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(state_summary, use_container_width=True)

# -------------------- RISK CLASSIFICATION --------------------

# -------------------- RISK CLASSIFICATION --------------------

elif page == "Risk Classification":
    st.header("District Crime-Risk Classification")

    years = sorted(district_data["Year"].unique())
    selected_year = st.selectbox(
        "Select year",
        years,
        index=len(years) - 1
    )

    risk_data = district_data[
        district_data["Year"] == selected_year
    ].copy()

    low_medium = risk_data["Crime_Rate_per_100k"].quantile(1 / 3)
    medium_high = risk_data["Crime_Rate_per_100k"].quantile(2 / 3)

    def classify(rate):
        if rate <= low_medium:
            return "Low"
        elif rate <= medium_high:
            return "Medium"
        return "High"

    risk_data["Crime_Risk"] = (
        risk_data["Crime_Rate_per_100k"].apply(classify)
    )

    counts = risk_data["Crime_Risk"].value_counts().reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={
            "x": "Risk Level",
            "y": "Number of Districts"
        },
        title=f"District Risk Distribution — {selected_year}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        risk_data.sort_values(
            "Crime_Rate_per_100k",
            ascending=False
        ),
        use_container_width=True
    )

    st.caption(
        "Risk levels use year-specific tertiles. They are relative "
        "categories within this dataset, not official police risk ratings."
    )


# -------------------- HOTSPOT ANALYSIS --------------------


elif page == "Hotspot Analysis":
    st.header("District Hotspot Ranking")

    # ---------- Year Filter ----------
    available_years = sorted(
        district_data["Year"].dropna().astype(int).unique().tolist()
    )

    selected_year = st.selectbox(
        "Select Year",
        available_years,
        index=len(available_years) - 1,
        key="hotspot_year_filter"
    )

    # Filter data by selected year
    year_data = district_data[
        district_data["Year"].astype(int) == selected_year
    ].copy()

    if not year_data.empty:
        st.caption(
            f"Ranking based on recorded crime rates for {selected_year}. "
            "This is not an official police risk rating."
        )

        st.subheader("Explore District Hotspots")

        # ---------- Hotspot Filters ----------
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            available_states = ["All"] + sorted(
                year_data["State"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_state = st.selectbox(
                "Filter by State",
                available_states,
                key="hotspot_state_filter"
            )

        with filter_col2:
            top_n = st.slider(
                "Number of districts",
                min_value=5,
                max_value=30,
                value=10,
                key="hotspot_top_n"
            )

        # Apply selected state
        filtered_hotspots = year_data.copy()

        if selected_state != "All":
            filtered_hotspots = filtered_hotspots[
                filtered_hotspots["State"].astype(str) == selected_state
            ]

        if filtered_hotspots.empty:
            st.info("No districts found for this selection.")

        else:
            # Rank districts for the selected year and state
            top = filtered_hotspots.nlargest(
                min(top_n, len(filtered_hotspots)),
                "Crime_Rate_per_100k"
            ).sort_values(
                "Crime_Rate_per_100k",
                ascending=True
            )

            fig = px.bar(
                top,
                x="Crime_Rate_per_100k",
                y="District",
                color="State",
                orientation="h",
                title=f"Districts Ranked by Crime Rate — {selected_year}",
                labels={
                    "Crime_Rate_per_100k": "Crime Rate per 100,000 People",
                    "District": "District"
                },
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#111111",
                plot_bgcolor="#171717",
                font_color="white",
                height=max(400, len(top) * 35)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                filtered_hotspots.sort_values(
                    "Crime_Rate_per_100k",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning("No district data is available for the selected year.")


        # ---------- Apply Filters ----------
        filtered_hotspots = hotspot_data.copy()

        if selected_state != "All":
            filtered_hotspots = filtered_hotspots[
                filtered_hotspots["State"].astype(str) == selected_state
            ]

        if filtered_hotspots.empty:
            st.info("No districts available for this selection.")

        else:
            top = (
                filtered_hotspots
                .sort_values(
                    "Crime_Rate_per_100k",
                    ascending=False
                )
                .head(top_n)
                .sort_values(
                    "Crime_Rate_per_100k",
                    ascending=True
                )
            )

            # ---------- Ranking Chart ----------
            fig = px.bar(
                top,
                x="Crime_Rate_per_100k",
                y="District",
                color="State",
                orientation="h",
                hover_data=[
                    "State",
                    "Crime_Rate_per_100k"
                ],
                title="Districts Ranked by Crime Rate",
                labels={
                    "Crime_Rate_per_100k": (
                        "Crime Rate per 100,000 People"
                    ),
                    "District": "District"
                },
                template="plotly_dark"
            )

            fig.update_layout(
                height=max(420, len(top) * 42),
                paper_bgcolor="#111111",
                plot_bgcolor="#171717",
                font_color="white",
                legend_title_text="State",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                f"Showing {len(top)} district(s) ranked "
                "by recorded crime rate."
            )

            st.subheader("District Ranking Data")

            st.dataframe(
                filtered_hotspots.sort_values(
                    "Crime_Rate_per_100k",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )
            # ---------- Geographic Hotspot Map ----------
   # -------------------- RISK CLASSIFICATION --------------------

elif page == "Risk Classification":
    st.header("District Crime-Risk Classification")

    years = sorted(district_data["Year"].unique())
    selected_year = st.selectbox(
        "Select year", years, index=len(years) - 1
    )

    risk_data = district_data[
        district_data["Year"] == selected_year
    ].copy()

    low_medium = risk_data["Crime_Rate_per_100k"].quantile(1 / 3)
    medium_high = risk_data["Crime_Rate_per_100k"].quantile(2 / 3)

    def classify(rate):
        if rate <= low_medium:
            return "Low"
        elif rate <= medium_high:
            return "Medium"
        return "High"

    risk_data["Crime_Risk"] = (
        risk_data["Crime_Rate_per_100k"].apply(classify)
    )

    counts = risk_data["Crime_Risk"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={
            "x": "Risk Level",
            "y": "Number of Districts"
        },
        title=f"District Risk Distribution — {selected_year}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        risk_data.sort_values(
            "Crime_Rate_per_100k", ascending=False
        ),
        use_container_width=True
    )

    st.caption(
        "Risk levels use year-specific tertiles. They are relative "
        "categories within this dataset, not official police risk ratings."
    )


# -------------------- HOTSPOT ANALYSIS --------------------

elif page == "Hotspot Analysis":
    st.header("District Hotspot Ranking")

    if not hotspot_data.empty:
        st.caption(
            "Ranking based on recorded crime rate. Geographic matches "
            "are shown only where district names match uniquely."
        )

        st.subheader("Explore District Hotspots")

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            available_states = ["All"] + sorted(
                hotspot_data["State"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_state = st.selectbox(
                "Filter by State",
                available_states,
                key="hotspot_state_filter"
            )

        with filter_col2:
            top_n = st.slider(
                "Number of districts",
                min_value=5,
                max_value=30,
                value=10,
                key="hotspot_top_n"
            )

        filtered_hotspots = hotspot_data.copy()

        if selected_state != "All":
            filtered_hotspots = filtered_hotspots[
                filtered_hotspots["State"].astype(str) == selected_state
            ]

        if filtered_hotspots.empty:
            st.info("No districts available for this selection.")

        else:
            top = (
                filtered_hotspots
                .sort_values("Crime_Rate_per_100k", ascending=False)
                .head(top_n)
                .sort_values("Crime_Rate_per_100k", ascending=True)
            )

            fig = px.bar(
                top,
                x="Crime_Rate_per_100k",
                y="District",
                color="State",
                orientation="h",
                hover_data=["State", "Crime_Rate_per_100k"],
                title="Districts Ranked by Crime Rate",
                labels={
                    "Crime_Rate_per_100k": (
                        "Crime Rate per 100,000 People"
                    ),
                    "District": "District"
                },
                template="plotly_dark"
            )

            fig.update_layout(
                height=max(420, len(top) * 42),
                paper_bgcolor="#111111",
                plot_bgcolor="#171717",
                font_color="white",
                legend_title_text="State",
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Showing {len(top)} district(s) ranked "
                "by recorded crime rate."
            )

            st.subheader("District Ranking Data")
            st.dataframe(
                filtered_hotspots.sort_values(
                    "Crime_Rate_per_100k", ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            # ---------- Geographic Hotspot Map ----------
           # -------------------- HOTSPOT ANALYSIS --------------------

elif page == "Hotspot Analysis":
    import json
    import re
    import unicodedata

    st.header("District Hotspot Analysis")
    st.caption(
        "Explore recorded district crime rates by year, state, and district. "
        "Only uniquely matched geographic boundaries are displayed."
    )

    # --------------------------------------------------
    # 1. YEAR FILTER
    # --------------------------------------------------
    available_years = sorted(
        district_data["Year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not available_years:
        st.warning("No year data is available.")

    else:
        filter1, filter2, filter3 = st.columns(3)

        with filter1:
            selected_year = st.selectbox(
                "Select Year",
                available_years,
                index=len(available_years) - 1,
                key="hotspot_year_filter"
            )

        # Filter district data by year
        year_data = district_data[
            district_data["Year"].astype(int) == selected_year
        ].copy()

        # --------------------------------------------------
        # 2. STATE FILTER
        # --------------------------------------------------
        available_states = sorted(
            year_data["State"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with filter2:
            selected_state = st.selectbox(
                "Select State",
                ["All States"] + available_states,
                key="hotspot_state_filter"
            )

        filtered_hotspots = year_data.copy()

        if selected_state != "All States":
            filtered_hotspots = filtered_hotspots[
                filtered_hotspots["State"].astype(str) == selected_state
            ]

        # --------------------------------------------------
        # 3. DISTRICT FILTER
        # --------------------------------------------------
        available_districts = sorted(
            filtered_hotspots["District"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with filter3:
            selected_district = st.selectbox(
                "Select District",
                ["All Districts"] + available_districts,
                key="hotspot_district_filter"
            )

        if selected_district != "All Districts":
            filtered_hotspots = filtered_hotspots[
                filtered_hotspots["District"].astype(str)
                == selected_district
            ]

        # --------------------------------------------------
        # 4. EMPTY DATA CHECK
        # --------------------------------------------------
        if filtered_hotspots.empty:
            st.warning("No records found for the selected filters.")

        else:
            # --------------------------------------------------
            # 5. SUMMARY METRICS
            # --------------------------------------------------
            metric1, metric2, metric3 = st.columns(3)

            metric1.metric("Selected Year", selected_year)

            metric2.metric(
                "Districts Shown",
                filtered_hotspots["District"].nunique()
            )

            metric3.metric(
                "Average Crime Rate",
                f'{filtered_hotspots["Crime_Rate_per_100k"].mean():.2f}'
            )

            st.divider()

            # --------------------------------------------------
            # 6. NUMBER OF DISTRICTS
            # --------------------------------------------------
            top_n = st.slider(
                "Number of districts to display",
                min_value=5,
                max_value=30,
                value=10,
                key="hotspot_top_n"
            )

            top = filtered_hotspots.nlargest(
                min(top_n, len(filtered_hotspots)),
                "Crime_Rate_per_100k"
            ).sort_values(
                "Crime_Rate_per_100k",
                ascending=True
            )

            # --------------------------------------------------
            # 7. DISTRICT RANKING CHART
            # --------------------------------------------------
            st.subheader("Districts Ranked by Crime Rate")

            fig = px.bar(
                top,
                x="Crime_Rate_per_100k",
                y="District",
                color="State",
                orientation="h",
                hover_data=["State", "Year", "Population", "Total_Cases"],
                title=f"District Crime Rates — {selected_year}",
                labels={
                    "Crime_Rate_per_100k":
                        "Crime Rate per 100,000 People",
                    "District": "District",
                    "State": "State"
                },
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#111111",
                plot_bgcolor="#171717",
                font_color="white",
                height=max(400, len(top) * 40),
                margin=dict(l=20, r=20, t=60, b=30)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --------------------------------------------------
            # 8. INDIA DISTRICT HOTSPOT MAP
            # --------------------------------------------------
            st.subheader("India District Hotspot Map")

         geo_path = os.path.join(
    BASE_DIR,
    "india_district_boundaries.geojson.gz"
)
            if not os.path.exists(geo_path):
                st.warning(
                    "GeoJSON boundary file not found: " + geo_path
                )

            else:
                try:
                   import gzip

with gzip.open(
    geo_path, "rt", encoding="utf-8"
) as geo_file:
    geo_data = json.load(geo_file)

                    def normalize_district(name):
                        name = unicodedata.normalize("NFKD", str(name))
                        name = name.encode(
                            "ascii", "ignore"
                        ).decode("ascii")
                        name = name.lower().strip()
                        name = re.sub(r"\bdistrict\b", "", name)
                        return re.sub(r"[^a-z0-9]", "", name)

                    features = geo_data.get("features", [])

                    # Keep geographic names that occur only once
                    geo_names = [
                        normalize_district(
                            feature.get("properties", {}).get(
                                "shapeName", ""
                            )
                        )
                        for feature in features
                    ]

                    geo_counts = pd.Series(
                        geo_names
                    ).value_counts().to_dict()

                    unique_features = [
                        feature for feature in features
                        if feature.get("properties", {}).get("shapeName")
                        and geo_counts.get(
                            normalize_district(
                                feature["properties"]["shapeName"]
                            ),
                            0
                        ) == 1
                    ]

                    geo_lookup = {
                        normalize_district(
                            feature["properties"]["shapeName"]
                        ): feature["properties"]["shapeName"]
                        for feature in unique_features
                    }

                    # Use exactly the same filtered records as the chart
                    map_df = filtered_hotspots.copy()

                    map_df["_key"] = map_df["District"].map(
                        normalize_district
                    )

                    # Exclude ambiguous names in the selected records
                    data_counts = map_df["_key"].value_counts().to_dict()

                    map_df = map_df[
                        map_df["_key"].map(data_counts).eq(1)
                    ].copy()

                    map_df["Map_District"] = map_df["_key"].map(
                        geo_lookup
                    )

                    map_df = map_df.dropna(
                        subset=["Map_District"]
                    )

                    matched_names = set(map_df["Map_District"])

                    map_features = [
                        feature for feature in unique_features
                        if feature["properties"]["shapeName"]
                        in matched_names
                    ]

                    map_geojson = {
                        "type": "FeatureCollection",
                        "features": map_features
                    }

                    if map_df.empty:
                        st.warning(
                            "No unique geographic boundaries matched "
                            "these filters. Try selecting All States "
                            "and All Districts."
                        )

                    else:
                        fig_map = px.choropleth_mapbox(
                            map_df,
                            geojson=map_geojson,
                            locations="Map_District",
                            featureidkey="properties.shapeName",
                            color="Crime_Rate_per_100k",
                            hover_name="District",
                            hover_data={
                                "State": True,
                                "Year": True,
                                "Crime_Rate_per_100k": ":.2f",
                                "Map_District": False
                            },
                            color_continuous_scale=[
                                [0.0, "#252525"],
                                [0.35, "#7f1d1d"],
                                [0.7, "#dc2626"],
                                [1.0, "#ff4d4d"]
                            ],
                            mapbox_style="carto-darkmatter",
                            center={"lat": 22.5, "lon": 79.0},
                            zoom=3.2,
                            opacity=0.75,
                            labels={
                                "Crime_Rate_per_100k":
                                    "Crime Rate per 100,000"
                            },
                            title=(
                                "District-Level Recorded Crime Rate — "
                                f"{selected_year}"
                            )
                        )

                        fig_map.update_layout(
                            height=650,
                            margin=dict(l=0, r=0, t=50, b=0),
                            paper_bgcolor="#111111",
                            font_color="white"
                        )

                        st.plotly_chart(
                            fig_map,
                            use_container_width=True
                        )

                        st.caption(
                            f"Mapped {map_df['District'].nunique()} "
                            f"district(s) out of "
                            f"{filtered_hotspots['District'].nunique()} "
                            "district(s) in this selection. "
                            "Unmatched districts are excluded."
                        )

                except Exception as e:
                    st.error(
                        "Unable to render the district map. "
                        f"Details: {e}"
                    )

            st.divider()

            # --------------------------------------------------
            # 9. FILTERED DISTRICT TABLE
            # --------------------------------------------------
            st.subheader("District Crime Records")

            st.dataframe(
                filtered_hotspots.sort_values(
                    "Crime_Rate_per_100k",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Crime rates are based on the available dataset. "
                "These rankings are not official police risk ratings, "
                "and unmatched district boundaries are not mapped."
            )

# -------------------- MODEL PERFORMANCE --------------------

# -------------------- MODEL PERFORMANCE --------------------

elif page == "Model Performance":
    st.header("Model Performance")

    if not evaluation.empty:
        row = evaluation.iloc[0]
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("MAE", f'{row["MAE"]:.4f}')
        c2.metric("MSE", f'{row["MSE"]:.4f}')
        c3.metric("RMSE", f'{row["RMSE"]:.4f}')
        c4.metric("R² Score", f'{row["R2_Score"]:.4f}')

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "The model has moderate predictive performance. "
            "Individual predictions can have substantial errors."
        )

    else:
        st.warning("Model evaluation file not found.")


# -------------------- CRIME PREDICTION --------------------

elif page == "Crime Prediction":
    st.header("Crime Rate Prediction")
    st.caption("Select a district and crime type to forecast its crime rate.")

    if model is None:
        st.error("Saved model pipeline not found.")
    elif source_df.empty:
        st.error("Historical crime dataset is unavailable.")
    else:
        required = ["State", "District", "Crime_Type", "Year", "Crime_Rate_per_100k", "Cases_Reported", "Population"]
        missing = [c for c in required if c not in source_df.columns]
        if missing:
            st.error(f"Dataset is missing columns: {missing}")
        else:
            states = sorted(source_df["State"].dropna().astype(str).unique())
            state = st.selectbox("Select State", states, key="forecast_state")
            state_df = source_df[source_df["State"].astype(str) == state]
            districts = sorted(state_df["District"].dropna().astype(str).unique())
            district = st.selectbox("Select District", districts, key="forecast_district")
            district_df = state_df[state_df["District"].astype(str) == district]
            crimes = sorted(district_df["Crime_Type"].dropna().astype(str).unique())
            crime = st.selectbox("Select Crime Type", crimes, key="forecast_crime")
            latest_year = int(source_df["Year"].max())
            forecast_year = st.selectbox("Forecast Year", list(range(latest_year + 1, latest_year + 6)), key="forecast_year")

            if st.button("Predict Crime Rate", key="run_forecast"):
                history = source_df[(source_df["State"].astype(str) == state) & (source_df["District"].astype(str) == district) & (source_df["Crime_Type"].astype(str) == crime)].copy()
                history["Year"] = history["Year"].astype(int)
                history = history.sort_values("Year").drop_duplicates("Year", keep="last")
                rate_by_year = dict(zip(history["Year"], history["Crime_Rate_per_100k"].astype(float)))
                cases_by_year = dict(zip(history["Year"], history["Cases_Reported"].astype(float)))
                pop_by_year = dict(zip(history["Year"], history["Population"].astype(float)))

                try:
                    prediction = None
                    for year in range(latest_year + 1, int(forecast_year) + 1):
                        needed = [year - 1, year - 2, year - 3]
                        if not all(y in rate_by_year for y in needed):
                            st.error(f"Missing historical rates required to forecast {year}.")
                            prediction = None
                            break
                        features = pd.DataFrame([{
                            "State": state,
                            "District": district,
                            "Crime_Type": crime,
                            "Year": year,
                            "Previous_Year_Rate": rate_by_year[year - 1],
                            "Rate_2_Years_Ago": rate_by_year[year - 2],
                            "Previous_Year_Cases": cases_by_year[year - 1],
                            "Previous_Year_Population": pop_by_year[year - 1],
                            "Previous_3_Year_Avg_Rate": sum(rate_by_year[y] for y in needed) / 3
                        }])
                        prediction = max(0.0, float(model.predict(features)[0]))
                        rate_by_year[year] = prediction
                        cases_by_year[year] = cases_by_year[year - 1]
                        pop_by_year[year] = pop_by_year[year - 1]

                    if prediction is not None:
                        st.metric("Predicted Crime Rate per 100,000 People", f"{prediction:.2f}")
                        st.write(f"**District:** {district} | **State:** {state} | **Crime type:** {crime} | **Year:** {forecast_year}")
                        st.warning("This is a model estimate, not actual crime data. Future cases and population are carried forward from the previous year, so forecasts—especially farther ahead—have additional uncertainty.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


