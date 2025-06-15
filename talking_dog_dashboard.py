#!/usr/bin/env python3
"""
🐕 TALKING DOG DASHBOARD - Alternative Data Signals Visualizer
Educational Demo Platform for Black Swan Detection

⚠️  EDUCATIONAL/DEMO PURPOSE ONLY - NO REAL TRADING ⚠️
🎯 Shows live alternative data signals and AI analysis
📊 Visual dashboard for signal detection and analysis
🧠 AI-powered context validation and confidence scoring
📈 Demo trading simulation for educational purposes

Author: Robert H.
Version: 1.1 - Fixed Decimal Strikes & Enhanced Forex
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import time
import requests
import json
import random
from typing import Dict, List
import asyncio
import threading

# Page configuration
st.set_page_config(
    page_title="🐕 Talking Dog - Alternative Data Dashboard",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CST Timezone
CST = pytz.timezone('America/Chicago')

def get_cst_time():
    return datetime.now(CST)

def get_market_status():
    now = get_cst_time()
    hour = now.hour
    minute = now.minute
    weekday = now.weekday()
    
    stock_open = (weekday < 5 and 
                  ((hour == 8 and minute >= 30) or 
                   (9 <= hour <= 14) or 
                   (hour == 15 and minute == 0)))
    
    options_open = (weekday < 5 and 
                    ((hour == 8 and minute >= 30) or 
                     (9 <= hour <= 14) or 
                     (hour == 15 and minute <= 15)))
    
    is_weekend_break = (weekday == 5 and hour >= 16) or (weekday == 6) or (weekday == 0 and hour < 17)
    is_daily_break = (weekday < 5 and hour == 16)
    futures_open = not (is_weekend_break or is_daily_break)
    
    # Forex markets: Sunday 5 PM CST - Friday 5 PM CST (24/5)
    forex_open = not ((weekday == 5 and hour >= 17) or (weekday == 6) or (weekday == 0 and hour < 17))
    
    return {
        'cst_time': now,
        'stock_market_open': stock_open,
        'options_market_open': options_open, 
        'futures_market_open': futures_open,
        'forex_market_open': forex_open,
        'weekend': weekday >= 5,
        'session': get_session_name(now)
    }

def get_session_name(cst_time):
    hour = cst_time.hour
    weekday = cst_time.weekday()
    
    if weekday >= 5:
        return "weekend"
    elif 8 <= hour <= 15:
        return "regular_hours"
    elif 16 <= hour <= 20:
        return "after_hours" 
    else:
        return "overnight"

class DemoSignalGenerator:
    """Generate realistic demo signals for educational purposes"""
    
    def __init__(self):
        self.locations = {
            'Pentagon': {'lat': 38.8704, 'lng': -77.0560, 'type': 'government'},
            'White House': {'lat': 38.8977, 'lng': -77.0365, 'type': 'government'},
            'Fed Building': {'lat': 38.8921, 'lng': -77.0446, 'type': 'government'},
            'Capitol Hill': {'lat': 38.8899, 'lng': -77.0091, 'type': 'government'},
            'FDA Silver Spring': {'lat': 39.0458, 'lng': -76.9781, 'type': 'regulatory'},
            'Apple Cupertino': {'lat': 37.3349, 'lng': -122.0090, 'type': 'tech'},
            'Google Mountain View': {'lat': 37.4220, 'lng': -122.0841, 'type': 'tech'},
            'Tesla Fremont': {'lat': 37.4939, 'lng': -121.9359, 'type': 'tech'},
            'Pfizer NYC': {'lat': 40.7518, 'lng': -73.9754, 'type': 'pharma'},
            'Goldman Sachs': {'lat': 40.7157, 'lng': -74.0134, 'type': 'finance'},
        }
        
        # Major forex pairs with realistic spreads
        self.forex_pairs = {
            'EUR/USD': {'base': 'EUR', 'quote': 'USD', 'typical_price': 1.0800, 'pip_value': 0.0001},
            'GBP/USD': {'base': 'GBP', 'quote': 'USD', 'typical_price': 1.2700, 'pip_value': 0.0001},
            'USD/JPY': {'base': 'USD', 'quote': 'JPY', 'typical_price': 149.50, 'pip_value': 0.01},
            'USD/CHF': {'base': 'USD', 'quote': 'CHF', 'typical_price': 0.9100, 'pip_value': 0.0001},
            'AUD/USD': {'base': 'AUD', 'quote': 'USD', 'typical_price': 0.6650, 'pip_value': 0.0001},
            'USD/CAD': {'base': 'USD', 'quote': 'CAD', 'typical_price': 1.3500, 'pip_value': 0.0001},
            'NZD/USD': {'base': 'NZD', 'quote': 'USD', 'typical_price': 0.6100, 'pip_value': 0.0001},
        }
        
        self.signal_history = []
        self.demo_trades = []
        self.forex_trades = []
        self.total_pnl = 0
        
    def generate_pentagon_pizza_signals(self) -> List[Dict]:
        """Generate realistic Pentagon Pizza signals"""
        signals = []
        
        market_status = get_market_status()
        current_time = market_status['cst_time']
        
        for location, coords in self.locations.items():
            if coords['type'] in ['government', 'regulatory']:
                # Simulate restaurant activity with realistic patterns
                hour = current_time.hour
                weekday = current_time.weekday()
                
                # Expected activity based on time/day
                if 11 <= hour <= 14:
                    expected_activity = 0.75  # Lunch rush
                elif 17 <= hour <= 20:
                    expected_activity = 0.65  # Dinner
                elif hour >= 21 or hour <= 6:
                    expected_activity = 0.08  # Late night/early morning
                else:
                    expected_activity = 0.4   # Normal business
                
                if weekday >= 5:  # Weekend
                    expected_activity *= 0.15  # Much lower on weekends
                
                # Add some randomness and occasional spikes
                activity_variance = np.random.normal(0, 0.1)
                
                # Weekend spikes are more suspicious
                if weekday >= 5:
                    spike_chance = 0.3
                    if np.random.random() < spike_chance:
                        activity_variance += np.random.uniform(0.5, 2.0)
                
                current_activity = expected_activity + activity_variance
                current_activity = max(0.01, current_activity)  # Keep positive
                
                # Calculate anomaly score
                anomaly_score = (current_activity - expected_activity) / 0.15
                
                if anomaly_score > 2.0:  # Only report significant anomalies
                    # Simulate news context for this signal
                    news_context = self.simulate_news_context(location, weekday)
                    
                    # AI confidence based on news context
                    if news_context['has_scheduled_events']:
                        ai_confidence = np.random.uniform(0.1, 0.4)  # Low confidence
                        trading_decision = 'ignore'
                    else:
                        ai_confidence = np.random.uniform(0.6, 0.9)  # High confidence
                        trading_decision = 'trade' if ai_confidence > 0.7 else 'monitor'
                    
                    signal = {
                        'location': location,
                        'type': 'Pentagon Pizza',
                        'current_activity': current_activity,
                        'expected_activity': expected_activity,
                        'anomaly_score': anomaly_score,
                        'strength': min(1.0, anomaly_score / 10.0),
                        'timestamp': current_time,
                        'news_context': news_context,
                        'ai_confidence': ai_confidence,
                        'trading_decision': trading_decision,
                        'coordinates': coords
                    }
                    
                    signals.append(signal)
        
        return signals
    
    def simulate_news_context(self, location: str, weekday: int) -> Dict:
        """Simulate realistic news context"""
        if weekday == 5:  # Saturday
            return {
                'has_scheduled_events': True,
                'events': ['Large Events in D.C.'],
                'black_swan_probability': 0.15,
                'explanation': 'High activity likely due to scheduled event'
            }
        elif np.random.random() < 0.2:  # 20% chance of breaking news
            return {
                'has_scheduled_events': False,
                'events': ['Unscheduled emergency meeting reported'],
                'black_swan_probability': 0.8,
                'explanation': 'No scheduled events found - potential black swan'
            }
        else:
            return {
                'has_scheduled_events': False,
                'events': [],
                'black_swan_probability': 0.5,
                'explanation': 'Normal activity patterns observed'
            }
    
    def generate_other_signals(self) -> List[Dict]:
        """Generate other alternative data signals"""
        signals = []
        
        # Traffic anomalies
        if np.random.random() < 0.4:
            locations = ['Pfizer NYC', 'Goldman Sachs', 'Apple Cupertino']
            location = np.random.choice(locations)
            
            signals.append({
                'location': location,
                'type': 'Traffic Anomaly',
                'anomaly_score': np.random.uniform(2.5, 8.0),
                'strength': np.random.uniform(0.6, 1.0),
                'timestamp': get_cst_time(),
                'ai_confidence': np.random.uniform(0.5, 0.8),
                'trading_decision': 'trade' if np.random.random() > 0.3 else 'monitor',
                'coordinates': self.locations[location]
            })
        
        # Corporate jet movements
        if np.random.random() < 0.3:
            signals.append({
                'location': 'Various Airports',
                'type': 'Corporate Jets',
                'anomaly_score': np.random.uniform(3.0, 12.0),
                'strength': np.random.uniform(0.7, 1.0),
                'timestamp': get_cst_time(),
                'ai_confidence': np.random.uniform(0.6, 0.9),
                'trading_decision': 'trade',
                'coordinates': {'lat': 40.7128, 'lng': -74.0060}  # NYC area
            })
        
        # Reddit sentiment
        if np.random.random() < 0.5:
            signals.append({
                'location': 'Reddit Analysis',
                'type': 'Social Sentiment',
                'anomaly_score': np.random.uniform(2.0, 6.0),
                'strength': np.random.uniform(0.4, 0.8),
                'timestamp': get_cst_time(),
                'ai_confidence': np.random.uniform(0.3, 0.7),
                'trading_decision': 'monitor' if np.random.random() > 0.4 else 'trade',
                'coordinates': {'lat': 37.7749, 'lng': -122.4194}  # SF
            })
        
        return signals
    
    def simulate_demo_trade(self, signal: Dict, ticker: str = 'SPY') -> Dict:
        """Simulate a realistic demo trade based on signal"""
        if signal['trading_decision'] != 'trade':
            return None
        
        # Get market status for realistic trade routing
        market_status = get_market_status()
        
        # Route to appropriate market based on what's open
        if market_status['options_market_open']:
            return self.simulate_options_trade(signal, ticker, market_status)
        elif market_status['forex_market_open']:
            return self.simulate_forex_trade(signal, market_status)
        else:
            return self.simulate_options_prep(signal, ticker, market_status)
    
    def simulate_options_trade(self, signal: Dict, ticker: str, market_status: Dict) -> Dict:
        """Simulate options trade when options market is open"""
        current_price = 500 + np.random.uniform(-10, 10)  # Simulate SPY price
        
        # Choose strike based on signal strength
        if signal['strength'] > 0.8:
            strike_offset = 0.02  # 2% OTM
        else:
            strike_offset = 0.05  # 5% OTM
        
        # Decide put or call based on signal type
        if signal['type'] == 'Pentagon Pizza':
            option_type = 'PUT'  # Government activity often bearish
            target_strike = current_price * (1 - strike_offset)
        else:
            option_type = np.random.choice(['PUT', 'CALL'])
            target_strike = current_price * (1 + strike_offset if option_type == 'CALL' else 1 - strike_offset)
        
        # CRITICAL FIX: Force whole dollar strikes (no decimals!)
        strike = int(round(target_strike))  # This ensures whole numbers only
        
        # Simulate realistic option price based on how far OTM
        otm_percent = abs(strike - current_price) / current_price
        if otm_percent < 0.02:  # Close to money
            option_price = np.random.uniform(1.50, 4.00)
        elif otm_percent < 0.05:  # Moderate OTM
            option_price = np.random.uniform(0.25, 1.50)
        else:  # Far OTM
            option_price = np.random.uniform(0.05, 0.50)
        
        quantity = max(1, int(signal['ai_confidence'] * 10))
        
        trade = {
            'timestamp': signal['timestamp'],
            'signal_type': signal['type'],
            'signal_location': signal['location'],
            'asset_type': 'OPTIONS',
            'ticker': ticker,
            'option_type': option_type,
            'strike': strike,  # Already int() converted - no decimals possible
            'option_price': round(option_price, 2),
            'quantity': quantity,
            'total_cost': round(option_price * quantity * 100, 2),
            'ai_confidence': signal['ai_confidence'],
            'status': '💰 OPTIONS BUY',
            'market_open': True,
            'pnl': 0
        }
        
        return trade
    
    def simulate_forex_trade(self, signal: Dict, market_status: Dict) -> Dict:
        """Simulate forex trade when forex market is open"""
        # Choose forex pair based on signal type and location
        if signal['type'] == 'Pentagon Pizza' or 'government' in signal.get('location', '').lower():
            # Government signals -> USD strength pairs
            pair_options = ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD']
            direction = 'SELL'  # Sell EUR/USD = Buy USD
        elif signal['type'] == 'Corporate Jets':
            # Corporate activity -> Major pairs
            pair_options = ['EUR/USD', 'GBP/USD', 'USD/JPY']
            direction = np.random.choice(['BUY', 'SELL'])
        else:
            # Other signals -> Random major pair
            pair_options = list(self.forex_pairs.keys())
            direction = np.random.choice(['BUY', 'SELL'])
        
        pair = np.random.choice(pair_options)
        pair_info = self.forex_pairs[pair]
        
        # Simulate current price with realistic spread
        base_price = pair_info['typical_price']
        spread = pair_info['pip_value'] * np.random.uniform(1, 3)  # 1-3 pip spread
        
        if direction == 'BUY':
            current_price = base_price * (1 + np.random.uniform(-0.01, 0.02)) + spread
        else:
            current_price = base_price * (1 + np.random.uniform(-0.02, 0.01)) - spread
        
        # Position size based on signal confidence (standard lot = 100,000 units)
        if signal['ai_confidence'] > 0.8:
            lot_size = 1.0  # Standard lot
        elif signal['ai_confidence'] > 0.6:
            lot_size = 0.5  # Half lot
        else:
            lot_size = 0.1  # Mini lot
        
        # Calculate position value (in USD)
        if 'USD' in pair_info['quote']:
            position_value = lot_size * 100000 * current_price
        else:
            position_value = lot_size * 100000  # For USD base pairs
        
        # Format price correctly based on pair
        if pair == 'USD/JPY':
            price_display = round(current_price, 2)  # 2 decimal places for JPY
        else:
            price_display = round(current_price, 4)  # 4 decimal places for others
        
        trade = {
            'timestamp': signal['timestamp'],
            'signal_type': signal['type'],
            'signal_location': signal['location'],
            'asset_type': 'FOREX',
            'pair': pair,
            'direction': direction,
            'lot_size': lot_size,
            'entry_price': price_display,
            'position_value': round(position_value, 2),
            'ai_confidence': signal['ai_confidence'],
            'status': f'💱 FOREX {direction}',
            'market_open': True,
            'pnl': np.random.uniform(-position_value * 0.02, position_value * 0.03)  # Realistic forex P&L
        }
        
        return trade
    
    def simulate_options_prep(self, signal: Dict, ticker: str, market_status: Dict) -> Dict:
        """Simulate options preparation when markets are closed - FIXED STRIKES"""
        current_price = 500 + np.random.uniform(-10, 10)
        
        if signal['strength'] > 0.8:
            strike_offset = 0.02
        else:
            strike_offset = 0.05
        
        if signal['type'] == 'Pentagon Pizza':
            option_type = 'PUT'
            target_strike = current_price * (1 - strike_offset)
        else:
            option_type = np.random.choice(['PUT', 'CALL'])
            target_strike = current_price * (1 + strike_offset if option_type == 'CALL' else 1 - strike_offset)
        
        # CRITICAL FIX: Force whole dollar strikes here too
        strike = int(round(target_strike))  # No decimals allowed
        
        option_price = np.random.uniform(0.25, 1.50)
        quantity = max(1, int(signal['ai_confidence'] * 10))
        
        trade = {
            'timestamp': signal['timestamp'],
            'signal_type': signal['type'],
            'signal_location': signal['location'],
            'asset_type': 'OPTIONS',
            'ticker': ticker,
            'option_type': option_type,
            'strike': strike,  # Already int() - guaranteed whole number
            'option_price': round(option_price, 2),
            'quantity': quantity,
            'total_cost': round(option_price * quantity * 100, 2),
            'ai_confidence': signal['ai_confidence'],
            'status': '📅 PREP FOR MONDAY',
            'market_open': False,
            'pnl': 0
        }
        
        return trade

# Initialize session state
if 'signal_generator' not in st.session_state:
    st.session_state.signal_generator = DemoSignalGenerator()
if 'last_update' not in st.session_state:
    st.session_state.last_update = get_cst_time()
if 'signals_history' not in st.session_state:
    st.session_state.signals_history = []
if 'demo_trades' not in st.session_state:
    st.session_state.demo_trades = []

# Main Dashboard
st.title("🐕 Talking Dog - Alternative Data Dashboard")
st.markdown("### 🎯 Educational Demo Platform for Black Swan Detection")

# Disclaimer
st.warning("⚠️ **EDUCATIONAL PURPOSE ONLY - NO REAL TRADING** ⚠️")
st.info(" **SIMULATED DATA:** This dashboard demonstrates alternative data concepts using simulated signals and demo trades.")

# Data Sources Explanation
with st.expander("📊 **Data Sources & Realism**"):
    st.markdown("""
    **SIMULATED (Educational Demo):**
    - Pentagon Pizza signals: Realistic patterns but simulated restaurant activity
    - Traffic anomalies: Simulated traffic data with realistic business hour patterns  
    - Reddit sentiment: Simulated post analysis
    - Corporate jet movements: Simulated flight data
    - Options prices: Simulated but realistic pricing models
    - Forex rates: Simulated but realistic major pair pricing
    
    **✅ REAL DATA:**
    - Market hours (CST timezone aware)
    - Actual trading session status
    - Proper market open/closed logic
    - Forex market hours (24/5 Sunday 5 PM - Friday 5 PM CST)
    - Smart trading routing (Options when open, Forex when options closed)
    - Realistic forex spreads and lot sizing
    
    **🎯 SMART ROUTING:** 
    - Options Market Open → Trade Options (whole dollar strikes)
    - Options Closed + Forex Open → Trade Forex (proper pip pricing)
    - All Markets Closed → Prepare Monday Options
    
    **🎯 PURPOSE:** 
    Shows how alternative data COULD work with proper real data feeds.
    Real implementation would use actual APIs (Google Places, FlightAware, etc.)
    """)

# Market Status with Better Context
market_status = get_market_status()
if not market_status['options_market_open'] and market_status['forex_market_open']:
    st.info(f"💱 **FOREX MODE ACTIVE** - Options closed, routing signals to forex pairs (24/5 trading)")
elif not market_status['options_market_open']:
    st.warning(f"📅 **ALL MARKETS CLOSED** - Any 'trades' shown are preparations for Monday market open")
elif market_status['weekend']:
    st.info(f"🌙 **WEEKEND MODE** - Increased sensitivity for unusual government activity")

# Sidebar Controls
st.sidebar.header("🎛️ Control Panel")

# Settings
st.sidebar.subheader("⚙️ Detection Settings")
min_signal_strength = st.sidebar.slider("Min Signal Strength (σ)", 1.0, 5.0, 2.5, 0.5)
ai_confidence_threshold = st.sidebar.slider("AI Confidence Threshold", 0.1, 0.9, 0.6, 0.1)
update_frequency = st.sidebar.slider("Update Frequency (seconds)", 10, 120, 30, 10)

# Manual refresh button
if st.sidebar.button("🔄 Manual Refresh"):
    st.session_state.last_update = get_cst_time() - timedelta(minutes=1)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)

# Market Status
market_status = get_market_status()
st.sidebar.subheader("📊 Market Status")
st.sidebar.write(f"🕐 **CST Time:** {market_status['cst_time'].strftime('%H:%M:%S %A')}")
st.sidebar.write(f"📅 **Session:** {market_status['session'].replace('_', ' ').title()}")

status_colors = {True: "🟢", False: "🔴"}
st.sidebar.write(f"{status_colors[market_status['stock_market_open']]} **Stock Market**")
st.sidebar.write(f"{status_colors[market_status['options_market_open']]} **Options Market**")
st.sidebar.write(f"{status_colors[market_status['futures_market_open']]} **Futures Market**")
st.sidebar.write(f"{status_colors[market_status['forex_market_open']]} **Forex Market** (24/5)")

# Trading routing info
st.sidebar.subheader("🎯 Auto Trading Route")
if market_status['options_market_open']:
    st.sidebar.success("→ Options Trading Active")
elif market_status['forex_market_open']:
    st.sidebar.info("→ Forex Trading Active")
else:
    st.sidebar.warning("→ Preparing Monday Options")

# Check if we need to update
current_time = get_cst_time()
if auto_refresh and (current_time - st.session_state.last_update).seconds >= update_frequency:
    # Generate new signals
    new_signals = st.session_state.signal_generator.generate_pentagon_pizza_signals()
    new_signals.extend(st.session_state.signal_generator.generate_other_signals())
    
    # Filter by strength threshold
    filtered_signals = [s for s in new_signals if s['anomaly_score'] >= min_signal_strength]
    
    # Add to history
    st.session_state.signals_history.extend(filtered_signals)
    
    # Generate demo trades for qualifying signals
    for signal in filtered_signals:
        if signal['ai_confidence'] >= ai_confidence_threshold:
            demo_trade = st.session_state.signal_generator.simulate_demo_trade(signal)
            if demo_trade:
                st.session_state.demo_trades.append(demo_trade)
    
    # Keep only last 50 signals and trades
    st.session_state.signals_history = st.session_state.signals_history[-50:]
    st.session_state.demo_trades = st.session_state.demo_trades[-20:]
    
    st.session_state.last_update = current_time

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("🎯 Live Signal Detection")
    
    # Current signals
    recent_signals = [s for s in st.session_state.signals_history if 
                     (current_time - s['timestamp']).seconds < 300]  # Last 5 minutes
    
    if recent_signals:
        for signal in recent_signals[-3:]:  # Show last 3
            confidence_color = "🟢" if signal['ai_confidence'] > 0.7 else "🟡" if signal['ai_confidence'] > 0.4 else "🔴"
            
            st.write(f"**{signal['type']} at {signal['location']}**")
            st.write(f"Strength: {signal['anomaly_score']:.1f}σ | AI Confidence: {confidence_color}{signal['ai_confidence']:.1%}")
            st.write(f"Decision: {signal['trading_decision'].upper()} | {signal['timestamp'].strftime('%H:%M:%S')}")
            st.write("---")
    else:
        st.write("🔍 Sniffing for signals... No anomalies detected in last 5 minutes")

with col2:
    st.subheader("🧠 AI Analysis")
    
    if recent_signals:
        avg_confidence = np.mean([s['ai_confidence'] for s in recent_signals])
        st.metric("Average AI Confidence", f"{avg_confidence:.1%}")
        
        trade_signals = len([s for s in recent_signals if s['trading_decision'] == 'trade'])
        st.metric("Trade Signals", trade_signals)
        
        # News context
        scheduled_events = sum(1 for s in recent_signals if s.get('news_context', {}).get('has_scheduled_events', False))
        st.metric("Scheduled Events", scheduled_events)
    else:
        st.metric("AI Status", "Analyzing...")

with col3:
    st.subheader("💰 Demo Trading")
    
    if st.session_state.demo_trades:
        # Calculate totals for different asset types
        options_trades = [t for t in st.session_state.demo_trades if t.get('asset_type') == 'OPTIONS']
        forex_trades = [t for t in st.session_state.demo_trades if t.get('asset_type') == 'FOREX']
        
        total_options_cost = sum(trade.get('total_cost', 0) for trade in options_trades)
        total_forex_value = sum(trade.get('position_value', 0) for trade in forex_trades)
        
        # Display metrics based on what's active
        if market_status['options_market_open']:
            st.metric("Options Investment", f"${total_options_cost:,.2f}")
            if total_options_cost > 0:
                simulated_pnl = np.random.uniform(-total_options_cost * 0.3, total_options_cost * 0.8)
                st.metric("Options P&L", f"${simulated_pnl:,.2f}", delta=f"{(simulated_pnl/total_options_cost)*100:.1f}%")
        
        elif market_status['forex_market_open']:
            st.metric("Forex Positions", f"${total_forex_value:,.2f}")
            if total_forex_value > 0:
                forex_pnl = sum(trade.get('pnl', 0) for trade in forex_trades)
                st.metric("Forex P&L", f"${forex_pnl:,.2f}", delta=f"{(forex_pnl/total_forex_value)*100:.2f}%")
        
        else:
            monday_prep = len([t for t in st.session_state.demo_trades if 'PREP FOR MONDAY' in t.get('status', '')])
            st.metric("Monday Prep Trades", monday_prep)
        
        # Total positions metric
        total_positions = len(options_trades) + len(forex_trades)
        st.metric("Total Positions", total_positions)
        
    else:
        if market_status['options_market_open']:
            st.metric("Status", "Ready for Options")
        elif market_status['forex_market_open']:
            st.metric("Status", "Ready for Forex")
        else:
            st.metric("Status", "Preparing for Monday")

# Signal strength chart
if st.session_state.signals_history:
    st.subheader("📊 Signal Strength Over Time")
    
    # Create DataFrame for plotting
    df = pd.DataFrame([
        {
            'time': s['timestamp'],
            'strength': s['anomaly_score'],
            'type': s['type'],
            'ai_confidence': s['ai_confidence']
        } for s in st.session_state.signals_history[-20:]
    ])
    
    if not df.empty:
        fig = px.scatter(df, x='time', y='strength', color='type', size='ai_confidence',
                        title="Signal Strength Detection")
        fig.add_hline(y=min_signal_strength, line_dash="dash", line_color="red",
                     annotation_text="Detection Threshold")
        st.plotly_chart(fig, use_container_width=True)

# Live map (if we have location signals)
location_signals = [s for s in recent_signals if 'coordinates' in s]
if location_signals:
    st.subheader("🗺️ Signal Locations")
    
    map_data = pd.DataFrame([
        {
            'lat': s['coordinates']['lat'],
            'lon': s['coordinates']['lng'],
            'location': s['location'],
            'strength': s['anomaly_score'],
            'type': s['type']
        } for s in location_signals
    ])
    
    st.map(map_data, size='strength')

# Demo trades table
if st.session_state.demo_trades:
    st.subheader("🎯 Recent Demo Trades")
    
    trades_df = pd.DataFrame(st.session_state.demo_trades[-10:])
    trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%H:%M:%S')
    
    # Create display columns based on asset type
    display_rows = []
    for _, trade in trades_df.iterrows():
        if trade.get('asset_type') == 'FOREX':
            display_rows.append({
                'Time': trade['timestamp'],
                'Signal': trade['signal_type'],
                'Asset': trade.get('pair', 'N/A'),
                'Direction': trade.get('direction', 'N/A'),
                'Size': f"{trade.get('lot_size', 0)} lots",
                'Price': f"{trade.get('entry_price', 0):.4f}",
                'Value': f"${trade.get('position_value', 0):,.0f}",
                'P&L': f"${trade.get('pnl', 0):,.2f}",
                'Status': trade['status'],
                'AI Conf': f"{trade['ai_confidence']:.1%}"
            })
        else:  # Options
            # DISPLAY VERIFICATION: Strikes should show as whole numbers
            strike = trade.get('strike', 0)
            display_rows.append({
                'Time': trade['timestamp'],
                'Signal': trade['signal_type'],
                'Asset': trade.get('ticker', 'SPY'),
                'Direction': trade.get('option_type', 'N/A'),
                'Size': f"{trade.get('quantity', 0)} contracts",
                'Strike': f"${int(strike)}",  # Force integer display - no decimals!
                'Price': f"${trade.get('option_price', 0):.2f}",
                'Value': f"${trade.get('total_cost', 0):,.0f}",
                'Status': trade['status'],
                'AI Conf': f"{trade['ai_confidence']:.1%}"
            })
    
    display_df = pd.DataFrame(display_rows)
    
    # Color coding function
    def color_row(row):
        if 'FOREX' in row.get('Status', ''):
            return ['background-color: #E3F2FD'] * len(row)  # Blue for forex
        elif 'PREP FOR MONDAY' in row.get('Status', ''):
            return ['background-color: #FFF3CD'] * len(row)  # Yellow for prep
        elif 'OPTIONS BUY' in row.get('Status', ''):
            return ['background-color: #D4EDDA'] * len(row)  # Green for options
        return [''] * len(row)
    
    # Apply styling
    styled_df = display_df.style.apply(lambda row: color_row(row), axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Market status explanation
    col1, col2 = st.columns(2)
    with col1:
        if any('PREP FOR MONDAY' in trade['status'] for trade in st.session_state.demo_trades[-5:]):
            st.caption("📅 **PREP FOR MONDAY:** Options market closed - the orders will be entered Monday at 8:30 AM CST")
        if any(int(trade.get('strike', 0)) == trade.get('strike', 0) for trade in st.session_state.demo_trades[-5:] if trade.get('asset_type') == 'OPTIONS'):
            st.caption("✅ **FIXED:** All option strikes are now whole dollars (no decimals)")
    with col2:
        if any('FOREX' in trade.get('status', '') for trade in st.session_state.demo_trades[-5:]):
            st.caption("💱 **FOREX ACTIVE:** Currency markets trade 24/5 - live execution when options closed")

# Performance metrics
if len(st.session_state.signals_history) > 10:
    st.subheader("📈 Performance Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signals = len(st.session_state.signals_history)
        st.metric("Total Signals", total_signals)
    
    with col2:
        trade_rate = len([s for s in st.session_state.signals_history if s['trading_decision'] == 'trade']) / total_signals
        st.metric("Trade Rate", f"{trade_rate:.1%}")
    
    with col3:
        avg_strength = np.mean([s['anomaly_score'] for s in st.session_state.signals_history])
        st.metric("Avg Signal Strength", f"{avg_strength:.1f}σ")
    
    with col4:
        avg_ai_conf = np.mean([s['ai_confidence'] for s in st.session_state.signals_history])
        st.metric("Avg AI Confidence", f"{avg_ai_conf:.1%}")

# Footer
st.markdown("---")
st.markdown("🐕 **Talking Dog Dashboard** | Educational Demo Platform | No Real Trading")
st.markdown(f"Last Updated: {st.session_state.last_update.strftime('%H:%M:%S CST')}")
st.success("✅ **VERSION 1.1 FIXES:** Decimal strikes eliminated • Enhanced forex trading • Smart market routing")

# Auto-refresh
if auto_refresh:
    time.sleep(1)
    st.rerun()
