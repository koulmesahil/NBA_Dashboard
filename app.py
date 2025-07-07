import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from nba_api.stats.endpoints import shotchartdetail, commonplayerinfo, teamgamelog
from nba_api.stats.static import players, teams
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import base64




import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch








# Page configuration
st.set_page_config(
    page_title="NBA Player Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: black;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50, #34495E);
    }
    
    .stSelectbox > div > div {
        background-color: inherit;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600)
def get_all_players():
    """Get all NBA players"""
    return players.get_players()

@st.cache_data(ttl=3600)
def get_player_info(player_id):
    """Get player information"""
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        return player_info.get_data_frames()[0]
    except Exception as e:
        st.error(f"Error fetching player info: {e}")
        return None

@st.cache_data(ttl=3600)
def get_shot_chart_data(player_id, season, season_type='Regular Season'):
    """Get shot chart data for a player"""
    try:
        shot_chart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star=season_type,
            context_measure_simple='FGA'
        )
        
        shots_df = shot_chart.get_data_frames()[0]
        league_avg = shot_chart.get_data_frames()[1]
        
        return shots_df, league_avg
    except Exception as e:
        st.error(f"Error fetching shot data: {e}")
        return None, None

def create_court_plot():
    """Create NBA court plot with properly aligned image background"""
    fig = go.Figure()

    # Load your court image from file and encode it to base64 for plotly
    try:
        with open("bbcourt_edited.png", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            image_uri = "data:image/png;base64," + encoded

        # NBA API coordinate system alignment
        # Basket is at (0, 0), half-court line at y=470
        # Court width is 500 units (-250 to 250)
        # Court length is 470 units (0 to 470)
        
        fig.add_layout_image(
            dict(
                source=image_uri,
                xref="x",
                yref="y",
                x=-250,        # Left edge of court
                y=400,         # Top edge (half-court line)
                sizex=500,     # Court width
                sizey=500,     # Court length
                sizing="contain",
                opacity=0.8,   # Slightly transparent so shots show clearly
                layer="below"
            )
        )
    except FileNotFoundError:
        st.warning("Court image not found. Using basic court outline.")
        # Fallback: create basic court outline

    # Layout settings to match NBA API coordinates
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-280, 280],  # Slightly wider for padding
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-50, 500]    # Show area behind basket
        ),
        
        title={
            'text': "Shot Density Map",
            'x': 0.5,  # center title horizontally
            'y': 0.75,  # place title slightly below the top (1.0 is top edge)
            'xanchor': 'center',
            'yanchor': 'top'
        },

        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        width=600,
        height=600
    )

    return fig





def add_shots_to_court(fig, shots_df, color_by='SHOT_MADE_FLAG'):
    """Add shot data to court plot with proper coordinate scaling"""
    if shots_df is None or shots_df.empty:
        return fig
    
    # NBA API coordinates are already in the correct scale
    # No coordinate transformation needed if using standard NBA API data
    shots_df = shots_df.copy()
    
    # Filter out shots that are too far (likely errors in data)
    shots_df = shots_df[
        (shots_df['LOC_X'].abs() <= 300) & 
        (shots_df['LOC_Y'] >= -50) & 
        (shots_df['LOC_Y'] <= 500)
    ]
    
    # Color mapping
    if color_by == 'SHOT_MADE_FLAG':
        # Made shots: green, Missed shots: red
        colors = ['#FF4444' if x == 0 else '#44FF44' for x in shots_df['SHOT_MADE_FLAG']]
        symbols = ['x' if x == 0 else 'circle' for x in shots_df['SHOT_MADE_FLAG']]
        hover_text = [
            f"Shot: {'Made' if made else 'Missed'}<br>"
            f"Distance: {dist} ft<br>"
            f"Type: {shot_type}<br>"
            f"Quarter: {period}<br>"
            f"Time: {time_remaining:02d}:{time_seconds:02d}"
            for made, dist, shot_type, period, time_remaining, time_seconds in zip(
                shots_df['SHOT_MADE_FLAG'],
                shots_df['SHOT_DISTANCE'],
                shots_df['ACTION_TYPE'],
                shots_df['PERIOD'],
                shots_df['MINUTES_REMAINING'],
                shots_df['SECONDS_REMAINING']
            )
        ]
    else:
        # Color by shot distance
        colors = shots_df['SHOT_DISTANCE']
        symbols = ['circle'] * len(shots_df)
        hover_text = [
            f"Distance: {dist} ft<br>"
            f"Type: {shot_type}<br>"
            f"Shot: {'Made' if made else 'Missed'}"
            for dist, shot_type, made in zip(
                shots_df['SHOT_DISTANCE'], 
                shots_df['ACTION_TYPE'],
                shots_df['SHOT_MADE_FLAG']
            )
        ]
    
    # Add scatter plot for shots
    fig.add_trace(
        go.Scatter(
            x=shots_df['LOC_X'],
            y=shots_df['LOC_Y'],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                symbol=symbols if color_by == 'SHOT_MADE_FLAG' else 'circle',
                colorscale='RdYlBu_r' if color_by != 'SHOT_MADE_FLAG' else None,
                opacity=0.8,
                line=dict(width=2, color='white'),
                showscale=color_by != 'SHOT_MADE_FLAG',
                colorbar=dict(
                    title="Distance (ft)",
                    titleside="right"
                ) if color_by != 'SHOT_MADE_FLAG' else None
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Shots'
        )
    )
    
    return fig

def create_shot_chart_with_stats(shots_df, title="Shot Chart", player_name=""):
    """Create a comprehensive shot chart with statistics"""
    fig = create_court_plot()
    fig = add_shots_to_court(fig, shots_df)
    
    # Calculate statistics
    if shots_df is not None and not shots_df.empty:
        total_shots = len(shots_df)
        made_shots = shots_df['SHOT_MADE_FLAG'].sum()
        fg_percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
        
        # Calculate zone statistics
        three_pointers = shots_df[shots_df['SHOT_TYPE'] == '3PT Field Goal']
        three_made = three_pointers['SHOT_MADE_FLAG'].sum() if len(three_pointers) > 0 else 0
        three_pct = (three_made / len(three_pointers) * 100) if len(three_pointers) > 0 else 0
        
        # Add statistics as annotations
        stats_text = (
            f"Overall: {made_shots}/{total_shots} ({fg_percentage:.1f}%)<br>"
            f"3PT: {three_made}/{len(three_pointers)} ({three_pct:.1f}%)"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1,
            align="left"
        )
    
    # Update title
    full_title = f"{player_name} {title}" if player_name else title
    fig.update_layout(
        title=dict(
            text=full_title, 
            font=dict(size=16, color="white"),
            x=0.5
        )
    )
    
    return fig





def calculate_shooting_zones(shots_df):
    """Calculate shooting percentages by zone"""
    if shots_df is None or shots_df.empty:
        return pd.DataFrame()
    
    # Define zones based on distance and location
    def get_zone(row):
        distance = row['SHOT_DISTANCE']
        if distance <= 8:
            return "Paint"
        elif distance <= 16:
            return "Mid-Range"
        elif distance <= 23:
            return "Long Mid-Range"
        else:
            return "Three-Point"
    
    shots_df['ZONE'] = shots_df.apply(get_zone, axis=1)
    
    zone_stats = shots_df.groupby('ZONE').agg({
        'SHOT_MADE_FLAG': ['count', 'sum', 'mean']
    }).round(3)
    
    zone_stats.columns = ['Attempts', 'Makes', 'Percentage']
    zone_stats['Percentage'] = (zone_stats['Percentage'] * 100).round(1)
    
    return zone_stats.reset_index()



#AI summary helper functions

@st.cache_resource
def load_model():
    """Load and cache the DistilGPT2 model and tokenizer."""
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_performance_summary(avg_distance, three_pt_pct, fg_percentage, total_shots, player_name, season):
    """
    Generate a meaningful performance summary using DistilGPT2.
    
    Args:
        avg_distance (float): Average shot distance in feet
        three_pt_pct (float): Three-point shooting percentage
        fg_percentage (float): Field goal percentage
        total_shots (int): Total number of shots taken
        player_name (str): Name of the player
        season (str): Season (e.g., "2023-24")
    
    Returns:
        str: Generated performance summary
    """
    try:
        # Load model and tokenizer
        model, tokenizer = load_model()
        
        # Determine performance levels for context
        def get_performance_level(percentage):
            if percentage >= 50:
                return "excellent"
            elif percentage >= 45:
                return "good"
            elif percentage >= 40:
                return "solid"
            elif percentage >= 35:
                return "average"
            else:
                return "poor"
        
        def get_volume_level(shots):
            if shots >= 500:
                return "high volume"
            elif shots >= 200:
                return "moderate volume"
            elif shots >= 100:
                return "low volume"
            else:
                return "very low volume"
        
        def get_range_description(distance):
            if distance >= 20:
                return "long-range"
            elif distance >= 15:
                return "mid-range"
            elif distance >= 10:
                return "short mid-range"
            else:
                return "close-range"
        
        fg_level = get_performance_level(fg_percentage)
        three_pt_level = get_performance_level(three_pt_pct)
        volume_level = get_volume_level(total_shots)
        range_desc = get_range_description(avg_distance)
        
        # Create a well-crafted stats-oriented prompt
        prompt = f"""NBA Statistical Analysis Report:

Player: {player_name}
Season: {season}
Total Shot Attempts: {total_shots}
Field Goal Percentage: {fg_percentage*100:.1f}%
Three-Point Percentage: {three_pt_pct*100:.1f}%
Average Shot Distance: {avg_distance:.1f} feet

Statistical Context:
- NBA average field goal percentage is approximately 46.5%
- NBA average three-point percentage is approximately 35.8%
- Average shot distance indicates shot selection preference
- Shot volume reflects player's offensive role and usage rate

Performance Analysis: Based on these statistics, {player_name} in the {season} season exhibits"""
        
        # Tokenize the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,  # Add 100 tokens to the prompt
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones(inputs.shape, dtype=torch.long),
                no_repeat_ngram_size=2,
                top_p=0.9
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated summary (after the prompt)
        summary_start = generated_text.find("Performance Analysis:")
        if summary_start != -1:
            summary = generated_text[summary_start:].replace("Performance Analysis:", "").strip()
            # Clean up and limit length
            sentences = summary.split('. ')
            if len(sentences) > 4:
                summary = '. '.join(sentences[:4]) + '.'
        else:
            # Fallback if pattern not found
            summary = generated_text[len(prompt):].strip()
            sentences = summary.split('. ')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:3]) + '.'
        
        # Ensure the summary is not empty and makes sense
        if len(summary.strip()) < 20:
            summary = create_fallback_summary(avg_distance, three_pt_pct, fg_percentage, total_shots, player_name)
        
        return summary.strip()
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return create_fallback_summary(avg_distance, three_pt_pct, fg_percentage, total_shots, player_name)

def create_fallback_summary(avg_distance, three_pt_pct, fg_percentage, total_shots, player_name):
    """Create a fallback summary if the model fails."""
    
    # Statistical comparisons to league averages
    fg_vs_avg = fg_percentage - 46.5  # NBA average FG%
    three_pt_vs_avg = three_pt_pct - 35.8  # NBA average 3P%
    
    # Performance assessments with specific stats
    if fg_percentage >= 50:
        fg_assessment = f"elite efficiency ({fg_percentage:.1f}%, {fg_vs_avg:+.1f}% above league average)"
    elif fg_percentage >= 46.5:
        fg_assessment = f"above-average efficiency ({fg_percentage:.1f}%, {fg_vs_avg:+.1f}% above league average)"
    elif fg_percentage >= 42:
        fg_assessment = f"solid efficiency ({fg_percentage:.1f}%, {fg_vs_avg:+.1f}% below league average)"
    else:
        fg_assessment = f"below-average efficiency ({fg_percentage:.1f}%, {fg_vs_avg:+.1f}% below league average)"
    
    if three_pt_pct >= 40:
        three_pt_assessment = f"elite three-point shooting ({three_pt_pct:.1f}%, {three_pt_vs_avg:+.1f}% above league average)"
    elif three_pt_pct >= 35.8:
        three_pt_assessment = f"above-average three-point shooting ({three_pt_pct:.1f}%, {three_pt_vs_avg:+.1f}% above league average)"
    elif three_pt_pct >= 30:
        three_pt_assessment = f"below-average three-point shooting ({three_pt_pct:.1f}%, {three_pt_vs_avg:+.1f}% below league average)"
    else:
        three_pt_assessment = f"poor three-point shooting ({three_pt_pct:.1f}%, {three_pt_vs_avg:+.1f}% below league average)"
    
    # Volume and distance analysis
    if total_shots >= 800:
        volume_analysis = f"high-volume scorer ({total_shots} attempts, indicating primary offensive role)"
    elif total_shots >= 400:
        volume_analysis = f"moderate-volume scorer ({total_shots} attempts, solid offensive contribution)"
    elif total_shots >= 200:
        volume_analysis = f"selective shooter ({total_shots} attempts, efficient shot selection)"
    else:
        volume_analysis = f"limited attempts ({total_shots} shots, likely role player)"
    
    if avg_distance >= 18:
        distance_analysis = f"perimeter-oriented player (avg: {avg_distance:.1f} ft, emphasizes long-range shots)"
    elif avg_distance >= 12:
        distance_analysis = f"balanced shot selection (avg: {avg_distance:.1f} ft, mix of perimeter and interior)"
    else:
        distance_analysis = f"interior-focused player (avg: {avg_distance:.1f} ft, emphasizes close-range shots)"
    
    return f"{player_name} demonstrates {fg_assessment} and {three_pt_assessment}. As a {volume_analysis}, he shows {distance_analysis}. These statistics indicate a player who averages {avg_distance:.1f} feet per shot with {total_shots} total attempts, shooting {fg_percentage:.1f}% overall and {three_pt_pct:.1f}% from three-point range."

# Integration function for your main app
def add_performance_summary_to_app(avg_distance, three_pt_pct, fg_percentage, total_shots, player_name, season):
    """
    Add the performance summary section to your Streamlit app.
    Call this function in your main() function where you want to display the summary.
    """
    st.markdown("### 🤖 AI Performance Analysis")
    
    with st.spinner("Generating AI analysis..."):
        summary = generate_performance_summary(
            avg_distance, three_pt_pct, fg_percentage, total_shots, player_name, season
        )
    
    # Display the summary in a nice container
    st.markdown(f"""
    <div style="
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        color: #2c3e50;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    ">
        <h4 style="margin-top: 0; color: #495057; font-size: 18px; margin-bottom: 15px; font-weight: 600;">
            AI Summary
        </h4>
        <div style="
            background: #ffffff;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            border-left: 3px solid #6c757d;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #6c757d; font-size: 14px;"><strong>Field Goal:</strong> {fg_percentage:.1f}%</span>
                <span style="color: #6c757d; font-size: 14px;"><strong>3-Point:</strong> {three_pt_pct:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #6c757d; font-size: 14px;"><strong>Total Shots:</strong> {total_shots}</span>
                <span style="color: #6c757d; font-size: 14px;"><strong>Avg Distance:</strong> {avg_distance:.1f} ft</span>
            </div>
        </div>
        <p style="margin-bottom: 0; line-height: 1.6; font-size: 15px; color: #495057;">
            {summary}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return summary







# Main App
def main():

    #Header

    st.markdown(
        """
        <h3 style='text-align: center; font-size: 40px;'>NBA Player Performance Dashboard</h3>
        """,
        unsafe_allow_html=True
    )

    
    st.sidebar.image("poster.png", width=600)


    st.sidebar.markdown("## 🎯 Player Selection")
    
    # Get all players
    all_players = get_all_players()
    player_names = [f"{p['full_name']}" for p in all_players]
    
    # Player selection
    selected_player_name = st.sidebar.selectbox(
        "Choose a Player",
        options=player_names,
        index=player_names.index("LeBron James") if "LeBron James" in player_names else 0
    )
    
    # Find selected player ID
    selected_player = next(p for p in all_players if p['full_name'] == selected_player_name)
    player_id = selected_player['id']
    
    # Season selection
    current_year = datetime.now().year
    seasons = [f"{year}-{str(year+1)[2:]}" for year in range(current_year-10, current_year)]
    selected_season = st.sidebar.selectbox("Season", seasons, index=len(seasons)-1)
    
    # Season type
    season_type = st.sidebar.selectbox("Season Type", ["Regular Season", "Playoffs"])
    
    # Filters
    st.sidebar.markdown("## 🎛️ Filters")
    
    # Load data
    with st.spinner("Loading shot data..."):
        shots_df, league_avg = get_shot_chart_data(player_id, selected_season, season_type)
    
    if shots_df is not None and not shots_df.empty:
        # Shot type filter
        shot_types = ["All"] + sorted(shots_df['ACTION_TYPE'].unique().tolist())
        selected_shot_type = st.sidebar.selectbox("Shot Type", shot_types)
        
        # Period filter
        periods = ["All"] + sorted(shots_df['PERIOD'].unique().tolist())
        selected_period = st.sidebar.selectbox("Period", periods)

        
        # Filter data
        filtered_shots = shots_df.copy()
        if selected_shot_type != "All":
            filtered_shots = filtered_shots[filtered_shots['ACTION_TYPE'] == selected_shot_type]
        if selected_period != "All":
            filtered_shots = filtered_shots[filtered_shots['PERIOD'] == selected_period]
        
        # Visualization options
        #st.sidebar.markdown("## 🎨 Visualization")
        color_option = "Make/Miss"




        
        
        # Main content area
        col1, col2 = st.columns([1, 2])
        

        
        with col1:
            # Player info
            player_info = get_player_info(player_id)
            total_shots = len(filtered_shots)
            made_shots = filtered_shots['SHOT_MADE_FLAG'].sum()
            fg_percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
                    
                    # Three-point stats
            three_pt_shots = filtered_shots[filtered_shots['SHOT_TYPE'] == '3PT Field Goal']
            three_pt_made = three_pt_shots['SHOT_MADE_FLAG'].sum() if not three_pt_shots.empty else 0
            three_pt_attempts = len(three_pt_shots)
            three_pt_pct = (three_pt_made / three_pt_attempts * 100) if three_pt_attempts > 0 else 0

            if player_info is not None:
                st.markdown("### Player Info")

                # Construct headshot image URL
                image_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

                st.markdown(f"""
                <div style="text-align: center;">
                    <img src="{image_url}" alt="{selected_player_name}" style="width: 250px; border-radius: 10px; margin-bottom: 20px;">
                    <div class="metric-container" style="display: inline-block; text-align: left;">
                        <h3 style="margin-bottom: 5px;">{selected_player_name}</h3>
                        <p><strong>Team:</strong> {player_info.iloc[0]['TEAM_NAME'] if not player_info.empty else 'N/A'}</p>
                        <p><strong>Position:</strong> {player_info.iloc[0]['POSITION'] if not player_info.empty else 'N/A'}</p>
                        <p><strong>Season:</strong> {selected_season}</p>
                        <hr style="margin: 10px 0;">
                        <p><strong>Field Goal:</strong> <span style="color: green; font-weight: bold;">{fg_percentage:.1f}%</span></p>
                        <p><strong>3-Point:</strong> <span style="color: blue; font-weight: bold;">{three_pt_pct:.1f}%</span></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)



        with col2:
            #st.markdown("### 🎯 Interactive Shot Chart")
            
            # Create court and add shots
            court_fig = create_court_plot()
            court_fig = add_shots_to_court(
                court_fig, 
                filtered_shots, 
                'SHOT_MADE_FLAG' 
            )
            
            # Display chart
            st.plotly_chart(court_fig, use_container_width=True)


        st.markdown("---")  # Add a separator line

        avg_distance = filtered_shots['SHOT_DISTANCE'].mean()
        three_pt_pct = (three_pt_made / three_pt_attempts * 100) if three_pt_attempts > 0 else 0
        fg_percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0
        total_shots = len(filtered_shots)


        # Generate and display AI summary
        add_performance_summary_to_app(
            avg_distance, three_pt_pct, fg_percentage, total_shots, 
            selected_player_name, selected_season)


            

        
        # Zone analysis
        st.markdown("### Shooting Zones Analysis")
        zone_stats = calculate_shooting_zones(filtered_shots)

        if not zone_stats.empty:
            # Create zone visualization
            fig_zones = go.Figure()
            
            fig_zones.add_trace(go.Pie(
                labels=zone_stats['ZONE'],
                values=zone_stats['Percentage'],
                text=[f"{pct}%<br>({makes}/{attempts})" 
                    for pct, makes, attempts in zip(zone_stats['Percentage'], 
                                                    zone_stats['Makes'], 
                                                    zone_stats['Attempts'])],
                textinfo='text+label',
                textposition='auto',
                marker=dict(
                    colors=['#FF6B35', '#F7931E', '#FFD23F', '#4ECDC4'],
                    line=dict(color='#FFFFFF', width=3)
                ),
                hovertemplate='<b>%{label}</b><br>' +
                            'Shooting %: %{value}%<br>' +
                            '<extra></extra>',
                pull=[0.05, 0.05, 0.05, 0.05],  # Slightly separate all slices
                rotation=45,
                name='Shooting %'
            ))
            
            fig_zones.update_layout(
                title={
                    'text': "Shooting Percentage by Zone",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': 'white'}
                },
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12)
                ),
                margin=dict(t=80, b=80, l=20, r=20)
            )
            
            # Create two columns for side-by-side layout
            col1, col2 = st.columns(2)  # Pie chart takes 2/3, table takes 1/3
            
            with col1:
                st.plotly_chart(fig_zones, use_container_width=True)
                #st.markdown("#### 📊 Zone Statistics")


            
            with col2:

                st.markdown("### ⚡ Quick Stats")

                if not filtered_shots.empty:
                    total_shots = len(filtered_shots)
                    made_shots = filtered_shots['SHOT_MADE_FLAG'].sum()
                    fg_percentage = (made_shots / total_shots * 100) if total_shots > 0 else 0

                    # Three-point stats
                    three_pt_shots = filtered_shots[filtered_shots['SHOT_TYPE'] == '3PT Field Goal']
                    three_pt_made = three_pt_shots['SHOT_MADE_FLAG'].sum() if not three_pt_shots.empty else 0
                    three_pt_attempts = len(three_pt_shots)
                    three_pt_pct = (three_pt_made / three_pt_attempts * 100) if three_pt_attempts > 0 else 0

                    avg_distance = filtered_shots['SHOT_DISTANCE'].mean()

                    # Create 3 columns for horizontal layout
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Field Goal %", f"{fg_percentage:.1f}%", f"{made_shots}/{total_shots}")

                    with col2:
                        st.metric("3-Point %", f"{three_pt_pct:.1f}%", f"{three_pt_made}/{three_pt_attempts}")

                    with col3:
                        st.metric("Avg Distance", f"{avg_distance:.1f} ft")


                #Zone Statistics
                #st.plotly_chart(fig_zones, use_container_width=True)
                st.markdown("#### 📊 Zone Statistics")

                def highlight_percentage_and_zone(row):
                    percentage = row['Percentage']
                    max_percentage = zone_stats['Percentage'].max()
                    min_percentage = zone_stats['Percentage'].min()
                    
                    styles = [''] * len(row)  # start with empty styles for all columns in this row
                    
                    if percentage == max_percentage:
                        # Highlight both ZONE and Percentage columns in green
                        styles[row.index.get_loc('ZONE')] = 'background-color: green; color: white; font-weight: bold'
                        styles[row.index.get_loc('Percentage')] = 'background-color: green; color: white; font-weight: bold'
                    elif percentage == min_percentage:
                        # Highlight both ZONE and Percentage columns in red
                        styles[row.index.get_loc('ZONE')] = 'background-color: red; color: white; font-weight: bold'
                        styles[row.index.get_loc('Percentage')] = 'background-color: red; color: white; font-weight: bold'
                    
                    return styles

                styled_df = zone_stats.style.apply(highlight_percentage_and_zone, axis=1)

                st.dataframe(styled_df, use_container_width=True)










                         


            
            
            
            
            
            # Shooting stats
            




    
    else:
        st.error("No shot data available for the selected player and season.")
        st.info("Try selecting a different player or season.")

if __name__ == "__main__":
    main()
