import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re

def clean_like_notebook(path_or_file) -> pd.DataFrame:
    raw = pd.read_csv(path_or_file, header=None)
    raw = raw.drop(index=range(0, 9)).reset_index(drop=True)
    raw.columns = raw.iloc[0]
    raw = raw.drop(index=0).reset_index(drop=True)
    return raw

# Load data
df = clean_like_notebook("M183891.csv")

# Get the label column (first column)
label_col = df.columns[0]
df[label_col] = df[label_col].astype(str).str.strip()

# Filter columns: keep only years >= 2020
def is_2020_plus(col):
    try:
        y = int(str(col).split()[0])
        return y >= 2020
    except Exception:
        return False

value_cols = [c for c in df.columns if is_2020_plus(c)]
df = df[[label_col] + value_cols]

# Normalize period labels
def normalise_period(c):
    s = str(c).strip()
    parts = s.split()
    if len(parts) != 2:
        return s
    y, q = parts
    q = q.upper().replace(" ", "")
    qnum = q.replace("Q", "")
    try:
        return f"{int(y)} Q{int(qnum)}"
    except:
        return s

df.columns = [label_col] + [normalise_period(c) for c in value_cols]

# Page config
st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Team 11's Product Demo")
st.markdown("---")

st.subheader("📊 Employment Changes Overview") 
st.markdown("---")
# ===== Define specific sectors with indentation for display =====
sectors_display = [("Total Changes In Employment", "Total Changes In Employment"),
    ("Goods Producing Industries", "Goods Producing Industries"),
    ("Manufacturing", "    ↳ Manufacturing"),
    ("Construction", "    ↳ Construction"),
    ("Others", "    ↳ Others"),
    ("Services Producing Industries", "Services Producing Industries"),
    ("Wholesale & Retail Trade", "    ↳ Wholesale & Retail Trade"),
    ("Transportation & Storage", "    ↳ Transportation & Storage"),
    ("Accommodation & Food Services", "    ↳ Accommodation & Food Services"),
    ("Information & Communications", "    ↳ Information & Communications"),
    ("Financial & Insurance Services", "    ↳ Financial & Insurance Services"),
    ("Real Estate, Professional Services And Administrative & Support Services", 
     "    ↳ Real Estate, Professional Services And Administrative & Support Services"),
    ("Public Administration & Education", "    ↳ Public Administration & Education"),
    ("Health & Social Services", "    ↳ Health & Social Services"),
    ("Arts, Entertainment & Recreation", "    ↳ Arts, Entertainment & Recreation"),
    ("Other Services - Others", "    ↳ Other Services - Others")
]

# Create mapping dictionaries
display_to_actual = {display: actual for actual, display in sectors_display}
actual_to_display = {actual: display for actual, display in sectors_display}

allowed_sectors = [actual for actual, display in sectors_display]

# Filter dataframe to only include allowed sectors
df_filtered = df[df['Data Series'].isin(allowed_sectors)]

# Get all available periods
all_periods = [c for c in df_filtered.columns if c != 'Data Series']
all_periods = sorted(all_periods,
                    key=lambda x: (int(x.split()[0]), int(x.split()[1].replace('Q',''))))

# ===== FILTERS =====
st.subheader("🔍 Filter Options")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_display = st.selectbox(
        "Select Industry Sector",
        [display for actual, display in sectors_display],
        help="Choose the industry sector to analyze"
    )
    selected_sector = display_to_actual[selected_display]

with col2:
    start_period = st.selectbox(
        "Start Period",
        all_periods,
        index=0,
        help="Select the starting quarter"
    )

with col3:
    end_period = st.selectbox(
        "End Period",
        all_periods,
        index=len(all_periods)-1,
        help="Select the ending quarter"
    )

st.markdown("---")

# Prepare data for the selected sector
row = df_filtered[df_filtered['Data Series'] == selected_sector]

# Filter columns based on selected time range
start_idx = all_periods.index(start_period)
end_idx = all_periods.index(end_period)

if start_idx > end_idx:
    st.error("⚠️ Start period must be before end period!")
else:
    cols_needed = all_periods[start_idx:end_idx+1]
    
    data = row[cols_needed].T
    data.columns = ['Total_Change']
    data = data.reset_index()
    data.columns = ['Period', 'Total_Change']
    data['Total_Change'] = pd.to_numeric(data['Total_Change'], errors='coerce')
    
    # Better colors with nice contrast against black background
    # Using vibrant blue for positive and orange for negative
    colors = ['#FF6B35' if x < 0 else '#118AB2' for x in data['Total_Change']]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['Period'],
        y=data['Total_Change'],
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=0.5,
        opacity=0.9,
        text=data['Total_Change'],  
        texttemplate='%{text:,.0f}',
        textposition='outside'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Employment Changes: {selected_sector}',
            'font': {'size': 20}
        },
        xaxis_title='Quarter',
        yaxis_title='Employment Change',
        height=600,
        showlegend=False,
        xaxis=dict(tickangle=0),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            tickformat=',',  
            separatethousands=True
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ===== CUMULATIVE LINE CHART SECTION =====
st.markdown("---")
st.subheader("📈 Cumulative Employment Changes Overview")
st.markdown("---")
# Custom CSS to change multiselect tag color
st.markdown("""
    <style>
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #4C78A8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Create long_df (same transformation as in your notebook)
long_df = df_filtered.melt(id_vars=[label_col], var_name="Period", value_name="Change")
long_df["Change"] = pd.to_numeric(long_df["Change"], errors="coerce")

# Extract Year and Quarter
def split_period(p):
    try:
        y_str, q_str = p.split()
        return int(y_str), int(q_str.replace("Q", ""))
    except:
        return np.nan, np.nan

year_q = long_df["Period"].apply(split_period)
long_df["Year"] = [t[0] for t in year_q]
long_df["Quarter"] = [t[1] for t in year_q]
long_df = long_df.dropna(subset=["Year", "Quarter"])
long_df = long_df[long_df["Year"] >= 2020]

# Filter options for line chart
st.subheader("🔍 Select Sectors to Compare")

col1_line, col2_line, col3_line = st.columns([2, 1, 1])

with col1_line:
    # Let user select multiple sectors
    selected_sectors_line = st.multiselect(
        "Choose sectors (select 2-5 for best visualisation)",
        allowed_sectors,
        default=["Goods Producing Industries", "Services Producing Industries"],
        help="Select multiple sectors to compare their cumulative employment changes"
    )

with col2_line:
    start_period_line = st.selectbox(
        "Start Period",
        all_periods,
        index=0,
        help="Select the starting quarter",
        key="start_line"
    )

with col3_line:
    end_period_line = st.selectbox(
        "End Period",
        all_periods,
        index=len(all_periods)-1,
        help="Select the ending quarter",
        key="end_line"
    )

st.markdown("---")

if len(selected_sectors_line) == 0:
    st.warning("⚠️ Please select at least one sector to display")
elif len(selected_sectors_line) > 5:
    st.warning("⚠️ Too many sectors selected. Please select 5 or fewer for better readability")
else:
    # Filter by time range
    start_idx_line = all_periods.index(start_period_line)
    end_idx_line = all_periods.index(end_period_line)
    
    if start_idx_line > end_idx_line:
        st.error("⚠️ Start period must be before end period!")
    else:
        # Prepare cumulative data
        cum_df = (long_df
                  .loc[long_df[label_col].isin(selected_sectors_line), 
                       [label_col, "Year", "Quarter", "Period", "Change"]]
                  .copy())
        
        cum_df["Change"] = pd.to_numeric(cum_df["Change"], errors="coerce").fillna(0)
        cum_df["__order"] = cum_df["Year"]*10 + cum_df["Quarter"]
        cum_df = cum_df.sort_values(["__order", label_col])
        
        # Filter by selected time range
        periods_needed = all_periods[start_idx_line:end_idx_line+1]
        cum_df = cum_df[cum_df["Period"].isin(periods_needed)]
        
        # Calculate cumulative sum
        cum_df["Cumulative"] = cum_df.groupby(label_col, sort=False)["Change"].cumsum()
        
        plot_df = cum_df[[label_col, "Period", "__order", "Cumulative"]].drop_duplicates()
        
        # Create line chart with Plotly
        fig2 = go.Figure()
        
        # Define color palette for ALL sectors
        color_palette = {
            "Total Changes In Employment": "#9D4EDD",
            "Goods Producing Industries": "#F28E2B",
            "Services Producing Industries": "#4C78A8",
            "Manufacturing": "#E15759",
            "Construction": "#76B7B2",
            "Others": "#AF7AA1",
            "Wholesale & Retail Trade": "#59A14F",
            "Transportation & Storage": "#EDC948",
            "Accommodation & Food Services": "#FF9DA7",
            "Information & Communications": "#B07AA1",
            "Financial & Insurance Services": "#F28EC2",
            "Real Estate, Professional Services And Administrative & Support Services": "#FFBE7D",
            "Public Administration & Education": "#8CD17D",
            "Health & Social Services": "#9C755F",
            "Arts, Entertainment & Recreation": "#BAB0AC",
            "Other Services - Others": "#86BCB6"
        }
        
        # Plot each sector
        for sector in selected_sectors_line:
            sub = plot_df[plot_df[label_col] == sector].sort_values("__order")
            color = color_palette.get(sector, "#FF006E")  # Bright pink as fallback
            
            fig2.add_trace(go.Scatter(
                x=sub["Period"],
                y=sub["Cumulative"],
                mode='lines+markers',
                name=sector,
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Cumulative: %{y:,.0f}<extra></extra>'
            ))
        
        # Update layout
        fig2.update_layout(
            title={
                'text': f'Cumulative Employment Change — Sector Comparison ({start_period_line} to {end_period_line})',
                'font': {'size': 20}
            },
            xaxis_title='Quarter',
            yaxis_title='Cumulative Employment Change',
            height=600,
            hovermode='x unified',
            xaxis=dict(tickangle=0),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                tickformat=',',
                separatethousands=True
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(color="black")
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)


# ===== SERVICES SUB-SECTOR CONTRIBUTION CHART =====
st.markdown("---")
st.subheader("📊 Services Sub-Sector Contributions")
st.markdown("---")

# Define the services sub-sectors
sectors_focus = [
    "Wholesale & Retail Trade",
    "Transportation & Storage",
    "Accommodation & Food Services",
    "Information & Communications",
    "Financial & Insurance Services",
    "Real Estate, Professional Services And Administrative & Support Services",
    "Public Administration & Education",
    "Health & Social Services",
    "Arts, Entertainment & Recreation",
    "Other Services - Others",
]

# Short display names for better readability
display_map = {
    "Wholesale & Retail Trade": "Wholesale & Retail",
    "Transportation & Storage": "Transport & Storage",
    "Accommodation & Food Services": "Accommodation & Food",
    "Information & Communications": "InfoComm",
    "Financial & Insurance Services": "Financial & Insurance",
    "Real Estate, Professional Services And Administrative & Support Services": "Real Estate / Prof / Admin",
    "Public Administration & Education": "Public Admin & Edu",
    "Health & Social Services": "Health & Social",
    "Arts, Entertainment & Recreation": "Arts & Recreation",
    "Other Services - Others": "Other Services",
}

# Filter options
st.subheader("🔍 Filter Options")
col1_contrib, col2_contrib, col3_contrib = st.columns([2, 1, 1])

with col1_contrib:
    selected_sectors_contrib = st.multiselect(
        "Choose services sub-sectors to analyze",
        sectors_focus,
        default=sectors_focus,  # All sectors selected by default
        help="Select which services sub-sectors to include in the analysis"
    )

with col2_contrib:
    start_period_contrib = st.selectbox(
        "Start Period",
        all_periods,
        index=0,
        help="Select the starting quarter",
        key="start_contrib"
    )

with col3_contrib:
    end_period_contrib = st.selectbox(
        "End Period",
        all_periods,
        index=len(all_periods)-1,
        help="Select the ending quarter",
        key="end_contrib"
    )

st.markdown("---")

# Check if user selected any sectors
if len(selected_sectors_contrib) == 0:
    st.warning("⚠️ Please select at least one sector to display")
else:
    # Filter by time range
    start_idx_contrib = all_periods.index(start_period_contrib)
    end_idx_contrib = all_periods.index(end_period_contrib)
    
    if start_idx_contrib > end_idx_contrib:
        st.error("⚠️ Start period must be before end period!")
    else:
        # Filter long_df by time range
        periods_contrib = all_periods[start_idx_contrib:end_idx_contrib+1]
        long_df_filtered = long_df[long_df["Period"].isin(periods_contrib)]
        
        # Filter to SELECTED services sub-sectors and aggregate
        sect_df = (long_df_filtered.loc[long_df_filtered[label_col].isin(selected_sectors_contrib), 
                                         [label_col, "Change"]]
                                    .rename(columns={label_col: "Sector"})
                                    .copy())
        sect_df["Change"] = pd.to_numeric(sect_df["Change"], errors="coerce").fillna(0)
        
        contrib = (sect_df.groupby("Sector", as_index=False)["Change"].sum()
                          .rename(columns={"Change": "Contribution"}))
        
        # Sort bars (largest to smallest)
        contrib = contrib.sort_values("Contribution", ascending=False).reset_index(drop=True)
        
        # Add display names
        contrib["Display_Name"] = contrib["Sector"].map(display_map)
        
        # Create colors based on positive/negative
        contrib["Color"] = contrib["Contribution"].apply(lambda x: "#4C78A8" if x >= 0 else "#E15759")
        
        # Create bar chart
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            x=contrib["Display_Name"],
            y=contrib["Contribution"],
            marker_color=contrib["Color"],
            marker_line_width=0,
            text=contrib["Contribution"],
            texttemplate='%{text:,.0f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        # Update layout
        fig3.update_layout(
            title={
                'text': f'Services Sub-Sector Contributions ({start_period_contrib} to {end_period_contrib})',
                'font': {'size': 20}
            },
            xaxis_title='',
            yaxis_title='Contribution to Services',
            height=600,
            showlegend=False,
            xaxis=dict(
                tickangle=0,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray',
                tickformat=',',
                separatethousands=True
            )
        )
        
        st.plotly_chart(fig3, use_container_width=True)

# ===== BUBBLE CHART: EMPLOYMENT VS VACANCIES COMPARISON =====
st.markdown("---")
st.subheader("💼 Services Sub-Sector - Employment vs Job Vacancies Comparison")
st.markdown("---")

# Load and clean employment data (annual data)
def clean_employment_data(path_or_file) -> pd.DataFrame:
    raw = pd.read_csv(path_or_file, header=None)
    raw = raw.drop(index=range(0, 10)).reset_index(drop=True)
    raw.columns = raw.iloc[0]
    raw = raw.drop(index=0).reset_index(drop=True)
    return raw

# Load and clean vacancy data (quarterly data)
def clean_vacancy_data(path_or_file) -> pd.DataFrame:
    raw = pd.read_csv(path_or_file, header=None)
    raw = raw.drop(index=range(0, 10)).reset_index(drop=True)
    raw.columns = raw.iloc[0]
    raw = raw.drop(index=0).reset_index(drop=True)
    return raw

# Load the datasets
df_employed = clean_employment_data("employed.csv")
df_vacancy = clean_vacancy_data("vacancy.csv")

# Define the target sectors we want (from vacancy file naming)
target_sectors_long = [
    "Wholesale & Retail Trade",
    "Transportation & Storage",
    "Accommodation & Food Services",
    "Information & Communications",
    "Financial & Insurance Services",
    "Real Estate, Professional Services And Administrative & Support Services",
    "Public Administration & Education",
    "Health & Social Services",
    "Arts, Entertainment & Recreation",
    "Other Services - Others",
]

# Short names for display
sector_short_names = {
    "Wholesale & Retail Trade": "Wholesale & Retail",
    "Transportation & Storage": "Transport & Storage",
    "Accommodation & Food Services": "Accommodation & Food",
    "Information & Communications": "InfoComm",
    "Financial & Insurance Services": "Financial & Insurance",
    "Real Estate, Professional Services And Administrative & Support Services": "Real Estate / Prof / Admin",
    "Public Administration & Education": "Public Admin & Edu",
    "Health & Social Services": "Health & Social",
    "Arts, Entertainment & Recreation": "Arts & Recreation",
    "Other Services - Others": "Other Services",
}

# ===== PROCESS EMPLOYMENT DATA =====
emp_label_col = df_employed.columns[0]
df_employed[emp_label_col] = df_employed[emp_label_col].astype(str).str.strip()

# Mapping for employment aggregation (some sectors need to be combined)
emp_sector_mapping = {
    "Wholesale & Retail Trade": "Wholesale & Retail Trade",
    "Transportation & Storage": "Transportation & Storage",
    "Accommodation & Food Services": "Accommodation & Food Services",
    "Information & Communications": "Information & Communications",
    "Financial & Insurance Services": "Financial & Insurance Services",
    # These need to be aggregated
    "Real Estate Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Professional Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Administrative & Support Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Public Administration & Education": "Public Administration & Education",
    "Health & Social Services": "Health & Social Services",
    "Arts, Entertainment & Recreation": "Arts, Entertainment & Recreation",
    "Other Services - Others": "Other Services - Others",
}

# Map employment sectors
df_employed['SectorLong'] = df_employed[emp_label_col].map(emp_sector_mapping)
df_employed = df_employed[df_employed['SectorLong'].notna()]

# Find year columns (they might have spaces or be integers)
all_cols = [str(c).strip() for c in df_employed.columns]
year_cols = []
for year in ['2024', '2023', '2022', '2021', '2020']:
    # Try to find column matching this year
    for col in df_employed.columns:
        if str(col).strip() == year:
            year_cols.append(col)
            break

# Convert year columns to numeric
for col in year_cols:
    df_employed[col] = pd.to_numeric(df_employed[col], errors='coerce')

# Aggregate employment by sector (in case there are sub-sectors to combine)
if year_cols:
    df_emp_final = df_employed.groupby('SectorLong', as_index=False)[year_cols].sum()
    df_emp_final = df_emp_final[df_emp_final['SectorLong'].isin(target_sectors_long)]
    
    # Rename columns to string years for consistency
    rename_dict = {col: str(col).strip() for col in year_cols}
    df_emp_final = df_emp_final.rename(columns=rename_dict)
else:
    st.error("⚠️ No year columns found in employment data")
    st.stop()

# ===== PROCESS VACANCY DATA =====
vac_label_col = df_vacancy.columns[0]
df_vacancy[vac_label_col] = df_vacancy[vac_label_col].astype(str).str.strip()

# Mapping for vacancy aggregation
vac_sector_mapping = {
    "Wholesale And Retail Trade": "Wholesale & Retail Trade",
    "Transportation And Storage": "Transportation & Storage",
    "Accommodation And Food Services": "Accommodation & Food Services",
    "Information And Communications": "Information & Communications",
    "Financial And Insurance Services": "Financial & Insurance Services",
    # These need to be aggregated
    "Real Estate Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Professional Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Administrative And Support Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Other Administrative & Support Services": "Real Estate, Professional Services And Administrative & Support Services",
    "Public Administration & Education": "Public Administration & Education",
    "Health & Social Services": "Health & Social Services",
    "Arts, Entertainment & Recreation": "Arts, Entertainment & Recreation",
    "Others": "Other Services - Others",
}

# Map vacancy sectors
df_vacancy['SectorLong'] = df_vacancy[vac_label_col].map(vac_sector_mapping)
df_vacancy = df_vacancy[df_vacancy['SectorLong'].notna()]

# Aggregate quarters to get annual totals for 2020-2024
def sum_quarters_for_year(row, year, all_cols):
    """Sum all quarters for a given year"""
    total = 0
    for col in all_cols:
        col_str = str(col).strip()
        # Match patterns like "2020 1Q", "2020 2Q", etc.
        if str(year) in col_str and any(q in col_str for q in ['1Q', '2Q', '3Q', '4Q']):
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                total += val
    return total

# Create annual vacancy totals with different column names to avoid conflict
for year in ['2020', '2021', '2022', '2023', '2024']:
    df_vacancy[f'Year_{year}'] = df_vacancy.apply(lambda row: sum_quarters_for_year(row, year, df_vacancy.columns), axis=1)

# Aggregate vacancy by sector
vac_year_cols = ['Year_2020', 'Year_2021', 'Year_2022', 'Year_2023', 'Year_2024']
df_vac_final = df_vacancy.groupby('SectorLong', as_index=False)[vac_year_cols].sum()
df_vac_final = df_vac_final[df_vac_final['SectorLong'].isin(target_sectors_long)]

# Rename vacancy columns to match employment (just '2020', '2021', etc.)
df_vac_final = df_vac_final.rename(columns={
    'Year_2020': '2020',
    'Year_2021': '2021',
    'Year_2022': '2022',
    'Year_2023': '2023',
    'Year_2024': '2024'
})

# ===== MERGE EMPLOYMENT AND VACANCY =====
df_merged = df_emp_final.merge(df_vac_final, on='SectorLong', how='inner', suffixes=('_emp', '_vac'))

# Add short names
df_merged['SectorShort'] = df_merged['SectorLong'].map(sector_short_names)

# ===== FILTERS =====
st.subheader("🔍 Filter Options")

col1_bubble, col2_bubble, col3_bubble = st.columns([2, 1, 1])

with col1_bubble:
    # Get all available sectors
    available_sectors = df_merged['SectorShort'].tolist()
    default_sectors = ['Accommodation & Food', 'Financial & Insurance', 'Health & Social']
    selected_sectors_bubble = st.multiselect(
        "Select Industries to Display",
        available_sectors,
        default=default_sectors,
        help="Choose which industries to show in the comparison",
        key="sectors_bubble"
    )

with col2_bubble:
    year1_bubble = st.selectbox(
        "Select First Year",
        ['2020', '2021', '2022', '2023', '2024'],
        index=0,
        help="Choose the first year to compare",
        key="year1_bubble"
    )

with col3_bubble:
    year2_bubble = st.selectbox(
        "Select Second Year",
        ['2020', '2021', '2022', '2023', '2024'],
        index=4,
        help="Choose the second year to compare",
        key="year2_bubble"
    )

st.markdown("---")

# ===== CREATE BUBBLE CHART =====
if not selected_sectors_bubble:
    st.warning("⚠️ Please select at least one industry to display")
elif year1_bubble == year2_bubble:
    st.warning("⚠️ Please select two different years for comparison")
else:
    # Filter merged data by selected sectors
    df_filtered = df_merged[df_merged['SectorShort'].isin(selected_sectors_bubble)]
    
    # Prepare data for plotting
    plot_data_list = []
    
    for year in [year1_bubble, year2_bubble]:
        year_data = df_filtered[['SectorShort', 'SectorLong']].copy()
        year_data['Employment'] = pd.to_numeric(df_filtered[f'{year}_emp'], errors='coerce')
        year_data['Vacancies'] = pd.to_numeric(df_filtered[f'{year}_vac'], errors='coerce')
        year_data['Tightness'] = (year_data['Vacancies'] / (year_data['Employment'] * 1000)) * 100
        year_data['Year'] = year
        plot_data_list.append(year_data)
    
    plot_data = pd.concat(plot_data_list, ignore_index=True)
    plot_data = plot_data.dropna(subset=['Employment', 'Vacancies'])
    
    # Create bubble chart
    fig4 = go.Figure()
    
    # Solid color scheme for years
    year_colors = {
        year1_bubble: '#FF6B35',  # Solid Orange
        year2_bubble: '#118AB2'   # Solid Blue
    }
    
    for year in [year1_bubble, year2_bubble]:
        year_df = plot_data[plot_data['Year'] == year]
        
        fig4.add_trace(go.Scatter(
            x=year_df['Employment'],
            y=year_df['Vacancies'],
            mode='markers+text',
            name=year,
            marker=dict(
                size=year_df['Tightness'] * 8,  # Scale bubble size by tightness
                color=year_colors[year],
                opacity=1.0,  
                line=dict(color='white', width=2),
                sizemode='diameter',
                sizemin=4
            ),
            text=year_df['SectorShort'],
            textposition='top center',
            textfont=dict(size=9, color='white'),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Employment: %{x:,.1f} persons<br>' +
                'Vacancies: %{y:,.0f}<br>' +
                'Tightness: %{customdata:.2f}%<br>' +
                '<extra></extra>'
            ),
            customdata=year_df['Tightness']
        ))
    
    # Update layout
    fig4.update_layout(
        title={
            'text': f'Services Sub-Sector Employment vs Job Vacancies: {year1_bubble} vs {year2_bubble}<br><sub>Bubble size represents labour market tightness (%)</sub>',
            'font': {'size': 20}
        },
        xaxis_title='Employment (Thousand Persons)',
        yaxis_title='Job Vacancies (Count)',
        height=650,
        hovermode='closest',
        xaxis=dict(
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
            tickformat=',',
            separatethousands=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
            tickformat=',',
            separatethousands=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(color="black", size=12)
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Add explanation
    st.caption(
        "💡 **How to read this chart:** Each bubble represents a sector. "
        "The X-axis shows employment (in thousands), Y-axis shows job vacancies (total annual count). "
        "Bubble size indicates labour market tightness (Vacancies ÷ Employment × 100%). "
        "Larger bubbles = tighter labour market for that sector."
    )


# ===== HEATMAP: AGE GROUP SHIFT BY INDUSTRY =====
st.markdown("---")
st.subheader("📊 Services Sub-Sector - Age Group Employment Shift")
st.markdown("---")

# Load age dataset
df_age = pd.read_csv("b.cleaned_industry_age_dashboard.csv")

# Ensure Year is numeric
df_age["Year"] = pd.to_numeric(df_age["Year"], errors="coerce").astype("Int64")

# Filter df_age to only include 2020 onwards
df_age_2020plus = df_age[df_age["Year"] >= 2020].copy()

# Filter options for heatmap
st.subheader("🔍 Filter Options")

col1_heat, col2_heat, col3_heat = st.columns([2, 2, 1])

with col1_heat:
    # Select industries
    industries_heat = st.multiselect(
        "Select Industries",
        sorted(df_age_2020plus["SectorShort"].unique()),
        default=df_age_2020plus["SectorShort"].unique(),  # All industries by default
        help="Choose which industries to include in the heatmap",
        key="industries_heat"
    )

with col2_heat:
    # Select age groups
    age_groups_heat = st.multiselect(
        "Select Age Groups",
        ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65+'],
        default=['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65+'],
        help="Choose which age groups to include in the heatmap",
        key="age_groups_heat"
    )

with col3_heat:
    st.write("")  # Spacing

col1_year, col2_year = st.columns([1, 1])

with col1_year:
    start_year_heat = st.selectbox(
        "Start Year",
        sorted(df_age_2020plus["Year"].unique()),
        index=0,
        help="Select the starting year",
        key="start_year_heat"
    )

with col2_year:
    end_year_heat = st.selectbox(
        "End Year",
        sorted(df_age_2020plus["Year"].unique()),
        index=len(df_age_2020plus["Year"].unique())-1,
        help="Select the ending year",
        key="end_year_heat"
    )

st.markdown("---")

# Check if selections are valid
if not industries_heat:
    st.warning("⚠️ Please select at least one industry")
elif not age_groups_heat:
    st.warning("⚠️ Please select at least one age group")
elif start_year_heat > end_year_heat:
    st.error("⚠️ Start year must be before or equal to end year!")
else:
    # Filter data by selected industries and age groups
    df_age_heat = df_age_2020plus[
        (df_age_2020plus["SectorShort"].isin(industries_heat)) & 
        (df_age_2020plus["AgeLabel"].isin(age_groups_heat))
    ].copy()
    
    # Calculate change between two years
    df_start = df_age_heat[df_age_heat["Year"] == start_year_heat].copy()
    df_end = df_age_heat[df_age_heat["Year"] == end_year_heat].copy()
    
    # Merge to calculate difference
    df_change = df_start.merge(
        df_end,
        on=["Sector", "SectorShort", "AgeLabel"],
        suffixes=("_start", "_end")
    )
    df_change["Change"] = df_change["Value_end"] - df_change["Value_start"]
    
    # Create pivot table for heatmap
    heatmap_data = df_change.pivot_table(
        index="SectorShort",
        columns="AgeLabel",
        values="Change",
        aggfunc="sum"
    )
    
    # Sort age groups properly (only include selected ones)
    age_order = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65+']
    existing_ages = [age for age in age_order if age in heatmap_data.columns]
    heatmap_data = heatmap_data[existing_ages]
    
    # Create heatmap
    fig5 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[
            [0.0, '#E15759'],    # Coral/red for most negative
            [0.2, '#F28E2B'],    # Orange
            [0.35, '#FFBE7D'],   # Light orange/peach
            [0.45, '#FFF4E6'],   # Very light peach
            [0.5, '#FFFFFF'],    # White at zero
            [0.55, '#E0F5F3'],   # Very light teal
            [0.65, '#B3E5DD'],   # Light teal
            [0.8, '#7CCABE'],    # Medium teal 
            [1.0, '#4C9B8E'] 
        ],
        text=heatmap_data.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title="Change (Thousands)",
            tickformat=',',
        ),
        hovertemplate='<b>%{y}</b><br>Age: %{x}<br>Change: %{z:.1f}<extra></extra>'
    ))
    
    fig5.update_layout(
        title={
            'text': f'Services Sub-Sector × Age — Shift from {start_year_heat} to {end_year_heat} (Thousands)',
            'font': {'size': 20}
        },
        xaxis_title='Age Group',
        yaxis_title='Services Sub-Secotr',
        height=600,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')  # Put first industry at top
    )
    
    st.plotly_chart(fig5, use_container_width=True)

# ===== OCCUPATION BY INDUSTRY ANALYSIS =====
st.markdown("---")
st.subheader("👔 Service Sub-Sector - Occupation Distribution")
st.markdown("---")

# ===========================
# Load and prepare occupation data
# ===========================
df_occupation = pd.read_csv("M182081.csv")

# Drop irrelevant rows and clean column names
df_occupation = df_occupation.drop(df_occupation.index[:9]).drop(df_occupation.index[171:])
df_occupation.columns = df_occupation.iloc[0].astype(str).str.strip()
df_occupation = df_occupation.drop(df_occupation.index[0])

# Function to split hierarchical text column into occupation / L1 / L2
def split_series_levels(df_in: pd.DataFrame) -> pd.DataFrame:
    s_raw = df_in['Data Series'].astype(str).str.replace("\t", "    ", regex=False)
    indent = s_raw.str.len() - s_raw.str.lstrip().str.len()
    name = s_raw.str.strip()

    occ_col, l1_col, l2_col = [], [], []
    cur_occ, cur_l1 = None, None
    for ind, nm in zip(indent, name):
        i = int(ind)
        if i == 0:
            cur_occ, cur_l1 = nm, None
            occ_col.append(cur_occ); l1_col.append(None); l2_col.append(None)
        elif i <= 2:
            cur_l1 = nm
            occ_col.append(cur_occ); l1_col.append(cur_l1); l2_col.append(None)
        else:
            occ_col.append(cur_occ); l1_col.append(cur_l1); l2_col.append(nm)

    out = df_in.copy()
    out.insert(0, "occupation", occ_col)
    out.insert(1, "industry_l1", l1_col)
    out.insert(2, "industry_l2", l2_col)
    return out

df_occupation = split_series_levels(df_occupation)

# Data cleaning and renaming
df_occupation = df_occupation[df_occupation["occupation"] != "All Occupation Groups, (Total Employed Residents)"].copy()
df_occupation = df_occupation[df_occupation["occupation"] != "Other Occupation Groups Nes"]

# Merge similar industries
merge_target_l2 = "Real Estate, Professional Services And Administrative & Support Services"
merge_list_l2 = ["Real Estate Services", "Professional Services", "Administrative & Support Services"]
df_occupation["industry_l2"] = df_occupation["industry_l2"].replace(merge_list_l2, merge_target_l2)

# Rename inconsistent labels
rename_map_l2 = {
    "Public Administration & Education Services": "Public Administration & Education",
    "Other Community, Social & Personal Services": "Other Services - Others",
}
df_occupation["industry_l2"] = df_occupation["industry_l2"].replace(rename_map_l2)

# Short display names
display_map_occ = {
    "Wholesale & Retail Trade": "Wholesale & Retail",
    "Transportation & Storage": "Transport & Storage",
    "Accommodation & Food Services": "Accommodation & Food",
    "Information & Communications": "InfoComm",
    "Financial & Insurance Services": "Financial & Insurance",
    "Real Estate, Professional Services And Administrative & Support Services": "Real Estate / Prof / Admin",
    "Public Administration & Education": "Public Admin & Edu",
    "Health & Social Services": "Health & Social",
    "Arts, Entertainment & Recreation": "Arts & Recreation",
    "Other Services - Others": "Other Services",
}
df_occupation["industry_l2"] = df_occupation["industry_l2"].replace(display_map_occ)

# Convert year columns to numeric
for y in [str(yr) for yr in range(2020, 2025)]:
    if y in df_occupation.columns:
        df_occupation[y] = pd.to_numeric(df_occupation[y], errors="coerce")

# Filter to L2 industries only
df_occ_l2 = df_occupation[df_occupation["industry_l2"].notna()].copy()

# ===========================
# FILTERS
# ===========================
st.subheader("🔍 Filter Options")

col1_occ, col2_occ, col3_occ, col4_occ = st.columns([2, 2, 1, 1])

with col1_occ:
    # Industry selection
    available_industries_occ = sorted(df_occ_l2["industry_l2"].unique())
    selected_industries_occ = st.multiselect(
        "Select Industries",
        available_industries_occ,
        default=['Accommodation & Food'],
        help="Choose which industries to analyze",
        key="industries_occ"
    )

with col2_occ:
    # Occupation selection
    available_occupations = sorted(df_occ_l2["occupation"].unique())
    selected_occupations_occ = st.multiselect(
        "Select Occupations",
        available_occupations,
        default=['Associate Professionals & Technicians', 'Professionals', 'Service & Sales Workers'],
        help="Choose which occupations to display",
        key="occupations_occ"
    )

with col3_occ:
    year1_occ = st.selectbox(
        "Select First Year",
        ['2020', '2021', '2022', '2023', '2024'],
        index=0,
        help="Choose the first year",
        key="year1_occ"
    )

with col4_occ:
    year2_occ = st.selectbox(
        "Select Second Year",
        ['2020', '2021', '2022', '2023', '2024'],
        index=4,
        help="Choose the second year",
        key="year2_occ"
    )

st.markdown("---")

# ===========================
# VISUALIZATIONS
# ===========================
if not selected_industries_occ:
    st.warning("⚠️ Please select at least one industry to display")
elif not selected_occupations_occ:
    st.warning("⚠️ Please select at least one occupation to display")
elif year1_occ == year2_occ:
    st.warning("⚠️ Please select two different years for comparison")
else:
    # Filter data by industries AND occupations
    df_filtered_occ = df_occ_l2[
        (df_occ_l2["industry_l2"].isin(selected_industries_occ)) &
        (df_occ_l2["occupation"].isin(selected_occupations_occ))
    ].copy()
    
    # ===== STACKED BAR CHART - Occupation Composition =====

    # Prepare data for stacked bar
    df_stacked = df_filtered_occ.melt(
        id_vars=['industry_l2', 'occupation'],
        value_vars=[year1_occ, year2_occ],
        var_name='Year',
        value_name='Employment'
    )
    df_stacked['Employment'] = pd.to_numeric(df_stacked['Employment'], errors='coerce')
    
    # Define color palette matching the occupation colors
    occupation_colors = {
        "Managers & Administrators (Including Working Proprietors)": "#4C78A8",
        "Professionals": "#5B8DBE",
        "Associate Professionals & Technicians": "#F28E2B",
        "Clerical Support Workers": "#E15759",
        "Service & Sales Workers": "#76B7B2",
        "Craftsmen & Related Trades Workers": "#59A14F",
        "Plant & Machine Operators & Assemblers": "#EDC948",
        "Cleaners, Labourers & Related Workers": "#FF9DA7"
    }
    
    # Create stacked bar chart
    fig6 = px.bar(
        df_stacked,
        x='industry_l2',
        y='Employment',
        color='occupation',
        facet_col='Year',
        title=f'Occupation Composition by Industry: {year1_occ} vs {year2_occ}',
        labels={'industry_l2': 'Industry', 'Employment': 'Employment (Thousands)', 'occupation': 'Occupation'},
        color_discrete_map=occupation_colors,
        category_orders={'occupation': sorted(df_stacked['occupation'].unique())},
        height=600
    )
    fig6.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig6.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig6, use_container_width=True)
    
    st.caption(
        "💡 **How to read this chart:** "
        "The stacked bars show the occupation composition of each industry. "
        "Each colour represents a different occupation group. "
        "Compare the two years to see how the occupation mix has changed."
    )

# ===== AGE DISTRIBUTION BY OCCUPATION ANALYSIS =====
st.markdown("---")
st.subheader("👥 Age Distribution by Occupation")
st.markdown("---")

# Load age and occupation data
df_age_occ = pd.read_csv("cleaned_age_and_occupation3.csv", header=1)

# Clean column names
df_age_occ.rename(columns={df_age_occ.columns[0]: 'Data Series'}, inplace=True)
df_age_occ.columns = df_age_occ.columns.str.strip()

# Remove rows with 'Others' and 'All Occupation Groups'
df_age_occ = df_age_occ[~df_age_occ['Data Series'].str.contains('Others|All Occupation Groups', case=False, na=False, regex=True)].copy()

# Select only years 2020-2024
years_list = ['2024', '2023', '2022', '2021', '2020']
columns_to_select = ['Data Series'] + years_list
df_age_occ = df_age_occ[columns_to_select].copy()

# Convert year columns to numeric
for year in years_list:
    df_age_occ[year] = pd.to_numeric(
        df_age_occ[year].astype(str).str.strip().replace({'na': 0, '-': 0}),
        errors='coerce'
    ).fillna(0)

# Transform hierarchical data into long format
import re

analysis_data = []
current_age_group = 'Unknown Age Group'

for _, row in df_age_occ.iterrows():
    data_series_raw = row['Data Series']
    is_occupation_row = data_series_raw.startswith(' ')
    clean_name = data_series_raw.lstrip()
    
    if not is_occupation_row:
        age_group_temp = clean_name.replace('Employed Residents ', '')
        current_age_group_raw = re.sub(r'\(.*?\)', '', age_group_temp).strip()
        current_age_group = current_age_group_raw.replace('Aged ', '')
    else:
        if current_age_group != 'Unknown Age Group':
            for year in years_list:
                analysis_data.append({
                    'Age Group': current_age_group,
                    'Occupation': clean_name,
                    'Year': int(year),
                    'Employment': row[year]
                })

df_age_occ_long = pd.DataFrame(analysis_data)

# Create proper age order
age_order = sorted(df_age_occ_long['Age Group'].unique())

# ===========================
# FILTERS
# ===========================
st.subheader("🔍 Filter Options")

col1_age, col2_age, col3_age = st.columns([2, 1, 1])

with col1_age:
    # Occupation selection
    available_occupations_age = sorted(df_age_occ_long['Occupation'].unique())
    selected_occupation_age = st.selectbox(
        "Select Occupation",
        available_occupations_age,
        index=available_occupations_age.index('Associate Professionals & Technicians'),
        help="Choose an occupation to analyze age distribution",
        key="occupation_age"
    )

with col2_age:
    year1_age = st.selectbox(
        "Select First Year",
        ['2020', '2021', '2022', '2023', '2024'],
        index=0,
        help="Choose the first year",
        key="year1_age"
    )

with col3_age:
    year2_age = st.selectbox(
        "Select Second Year",
        ['2020', '2021', '2022', '2023', '2024'],
        index=4,
        help="Choose the second year",
        key="year2_age"
    )

st.markdown("---")

# ===========================
# VISUALIZATIONS
# ===========================
if selected_occupation_age is None:
    st.warning("⚠️ Please select an occupation to display")
elif year1_age == year2_age:
    st.warning("⚠️ Please select two different years for comparison")
else:
    # Filter data for selected occupation and years
    df_filtered_age = df_age_occ_long[
        (df_age_occ_long['Occupation'] == selected_occupation_age) &
        (df_age_occ_long['Year'].isin([int(year1_age), int(year2_age)]))
    ].copy()
    
    # ===== Growth by Age Group (Horizontal Bar Chart) =====
    
    # Calculate growth
    df_growth_age = df_filtered_age.pivot_table(
        index='Age Group',
        columns='Year',
        values='Employment',
        fill_value=0
    )
    
    df_growth_age['Growth'] = df_growth_age[int(year2_age)] - df_growth_age[int(year1_age)]
    df_growth_age = df_growth_age.reset_index()
    
    # Sort by age group (ascending order)
    df_growth_age['Age Group'] = pd.Categorical(df_growth_age['Age Group'], categories=age_order, ordered=True)
    df_growth_age = df_growth_age.sort_values('Age Group', ascending=True)
    
    # Color based on positive/negative
    df_growth_age['Color'] = df_growth_age['Growth'].apply(
        lambda x: '#E15759' if x < 0 else '#4C78A8'
    )
    
    fig_growth_age = go.Figure()
    
    fig_growth_age.add_trace(go.Bar(
        x=df_growth_age['Growth'],
        y=df_growth_age['Age Group'],
        orientation='h',
        marker_color=df_growth_age['Color'],
        text=df_growth_age['Growth'],
        texttemplate='%{text:,.1f}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Growth: %{x:,.1f} thousands<extra></extra>'
    ))
    
    fig_growth_age.update_layout(
        title={
            'text': f'{selected_occupation_age}: Growth by Age Group from {year1_age} to {year2_age}',
            'font': {'size': 20}
        },
        xaxis_title='Employment Change (Thousands)',
        yaxis_title='Age Group',
        height=600,
        showlegend=False,
        yaxis={'categoryorder': 'array', 'categoryarray': list(df_growth_age['Age Group'])},
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            tickformat=',',
            separatethousands=True
        )
    )
    
    st.plotly_chart(fig_growth_age, use_container_width=True)
    
    st.caption(
        "💡 **How to read this chart:** "
        "This shows which age groups within the selected occupation grew or declined. "
        "Positive values (blue) indicate growth, while negative values (red) indicate decline. "
        "This helps identify aging workforce trends and recruitment patterns by age."

    )
