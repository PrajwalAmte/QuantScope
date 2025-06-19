import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import re
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def initialize_groq_llm():
    """Initialize Groq LLM with Llama - cached to avoid re-initialization"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None, "GROQ_API_KEY not found in environment variables"

    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.7,
            max_tokens=1500
        )
        return llm, None
    except Exception as e:
        return None, f"Failed to initialize Groq LLM: {str(e)}"


def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_symbol = None
        st.session_state.stock_data = {}
        st.session_state.chat_messages = {}
        st.session_state.llm_error = None


def get_stock_context_for_llm(symbol, metrics, data):
    """Prepare stock context for LLM"""
    if data is None or data.empty:
        return ""

    current_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

    current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not data['RSI'].isna().iloc[-1] else None
    sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns and not data['SMA_20'].isna().iloc[-1] else None
    sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns and not data['SMA_50'].isna().iloc[-1] else None

    rsi_str = f"{current_rsi:.1f}" if current_rsi is not None else "N/A"
    sma_20_str = f"{sma_20:.2f}" if sma_20 is not None else "N/A"
    sma_50_str = f"{sma_50:.2f}" if sma_50 is not None else "N/A"

    context = f"""
Current Stock Analysis Context for {symbol}:

Price Information:
- Current Price: {current_price:.2f}
- Previous Close: {previous_price:.2f}
- Price Change: {price_change:.2f} ({price_change_pct:.2f}%)
- Period High: {data['High'].max():.2f}
- Period Low: {data['Low'].min():.2f}
- Average Volume: {data['Volume'].mean():.0f}

Technical Indicators:
- RSI: {rsi_str}
- SMA 20: {sma_20_str}
- SMA 50: {sma_50_str}

Key Metrics:
{chr(10).join([f"- {k}: {v}" for k, v in metrics.items()])}

Recent Price Trend: {data['Close'].tail(5).tolist()}
"""
    return context


def chat_with_llm(user_question, stock_context):
    """Chat with LLM about the stock"""
    llm, error = initialize_groq_llm()

    if error:
        return f"LLM initialization error: {error}"

    if not llm:
        return "LLM is not available. Please check your GROQ_API_KEY."

    try:
        system_message = SystemMessage(content=f"""
You are a professional stock analysis assistant. You have access to the following stock data and context:

{stock_context}

Please provide helpful, accurate, and professional analysis based on this data. Focus on:
1. Technical analysis insights
2. Price trends and patterns
3. Risk assessment
4. Market signals

Keep responses concise but informative. Do not provide financial advice or investment recommendations.
""")

        human_message = HumanMessage(content=user_question)
        response = llm.invoke([system_message, human_message])
        return response.content

    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"


class StockAnalyzer:
    def __init__(self):
        self.market_suffixes = {
            'Indian': '.NS',
            'US': '',
            'UK': '.L',
            'Canada': '.TO',
            'Australia': '.AX',
            'Germany': '.DE',
            'France': '.PA',
            'Japan': '.T',
            'Hong Kong': '.HK'
        }

        self.market_currencies = {
            'Indian': '₹',
            'US': '$',
            'UK': '£',
            'Canada': 'C$',
            'Australia': 'A$',
            'Germany': '€',
            'France': '€',
            'Japan': '¥',
            'Hong Kong': 'HK$'
        }

    def format_symbol(self, symbol, market):
        """Format stock symbol based on market"""
        symbol = symbol.upper().strip()
        suffix = self.market_suffixes.get(market, '')

        for suf in self.market_suffixes.values():
            if symbol.endswith(suf):
                symbol = symbol.replace(suf, '')
                break

        return f"{symbol}{suffix}"

    def get_stock_symbol(self, query, market):
        """Extract stock symbol from query"""
        common_words = ['FOR', 'AND', 'THE', 'OF', 'TO', 'IN', 'ON', 'AT', 'BY', 'WITH']
        words = query.upper().split()

        for word in words:
            if re.match(r'^[A-Z0-9]{2,10}$', word) and word not in common_words:
                return self.format_symbol(word, market)
        return None

    def get_timeframe(self, query):
        """Extract timeframe from query"""
        query_lower = query.lower()

        timeframe_mapping = {
            'ytd': ['ytd', 'year to date'],
            '1y': ['1y', 'year', 'annual'],
            '6mo': ['6mo', '6 month', 'half year'],
            '3mo': ['3mo', '3 month', 'quarter'],
            '1mo': ['1mo', 'month'],
            '1d': ['1d', 'day', 'today']
        }

        for period, keywords in timeframe_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return period

        return '3mo'  # default

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_stock_data(_self, symbol, period):
        """Fetch stock data using yfinance with caching"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None, f"No data found for symbol: {symbol}"
            return data, None
        except Exception as e:
            return None, f"Error fetching data: {str(e)}"

    def create_price_chart(self, data, symbol, chart_type="line"):
        """Create interactive price chart"""
        fig = go.Figure()

        if chart_type == "candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{symbol} Close Price',
                line=dict(width=2, color='#1f77b4')
            ))

        fig.update_layout(
            title=f'{symbol} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=450,
            showlegend=False
        )

        return fig

    def create_volume_chart(self, data, symbol):
        """Create volume chart"""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name=f'{symbol} Volume',
            marker_color='#17becf'
        ))

        fig.update_layout(
            title=f'{symbol} Trading Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_white',
            height=350,
            showlegend=False
        )

        return fig

    def create_technical_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()

        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

        return data

    def create_technical_chart(self, data, symbol):
        """Create technical indicators chart"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Close Price', line=dict(color='#1f77b4', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            name='SMA 20', line=dict(color='#ff7f0e', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'],
            name='SMA 50', line=dict(color='#d62728', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'],
            name='BB Upper', line=dict(color='gray', dash='dot'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'],
            name='BB Lower', line=dict(color='gray', dash='dot'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ))

        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=450
        )

        return fig

    def create_rsi_chart(self, data, symbol):
        """Create RSI chart"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name='RSI', line=dict(color='#9467bd', width=2)
        ))

        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")

        fig.update_layout(
            title=f'{symbol} RSI (Relative Strength Index)',
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_white',
            height=300,
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )

        return fig

    def calculate_metrics(self, data, currency_symbol):
        """Calculate key financial metrics"""
        if data.empty:
            return {}

        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        high_price = data['High'].max()
        low_price = data['Low'].min()
        avg_volume = data['Volume'].mean()

        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * 100

        current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not data['RSI'].isna().iloc[-1] else None

        metrics = {
            'Current Price': f"{currency_symbol}{current_price:.2f}",
            'Period Change': f"{currency_symbol}{current_price - start_price:.2f}",
            'Period Change %': f"{((current_price - start_price) / start_price) * 100:.2f}%",
            'Period High': f"{currency_symbol}{high_price:.2f}",
            'Period Low': f"{currency_symbol}{low_price:.2f}",
            'Average Volume': f"{avg_volume:,.0f}",
            'Volatility': f"{volatility:.2f}%",
        }

        if current_rsi is not None:
            metrics['Current RSI'] = f"{current_rsi:.1f}"

        return metrics

    def generate_ai_summary(self, symbol, metrics, data):
        """Generate a simple AI-like summary based on stock metrics"""
        current = metrics.get("Current Price", "")
        change = metrics.get("Period Change", "")
        change_pct = metrics.get("Period Change %", "")
        rsi = metrics.get("Current RSI", "N/A")

        trend = "neutral"
        try:
            rsi_val = float(rsi)
            if rsi_val > 70:
                trend = "overbought"
            elif rsi_val < 30:
                trend = "oversold"
        except:
            trend = "neutral"

        volatility = metrics.get("Volatility", "N/A")
        try:
            vol_val = float(volatility.replace('%', ''))
            vol_level = "high" if vol_val > 2 else "low"
        except:
            vol_level = "moderate"

        return (
            f"The current price of **{symbol}** is {current}, with a net change of {change} "
            f"({change_pct}) over the selected period. The RSI value is {rsi}, suggesting a **{trend}** trend. "
            f"Volatility is {volatility}, indicating a **{vol_level}** fluctuation in price."
        )


def display_chat_interface(symbol):
    """Display improved chat interface without page refresh issues"""
    st.subheader(f"💬 Chat about {symbol}")

    # Check LLM availability
    llm, error = initialize_groq_llm()
    if error:
        st.error(f"AI Chat unavailable: {error}")
        st.info("Please set GROQ_API_KEY in your environment or .env file")
        return

    # Initialize chat messages for this symbol
    if symbol not in st.session_state.chat_messages:
        st.session_state.chat_messages[symbol] = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages[symbol]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input using st.chat_input (more stable than text_input + button)
    if prompt := st.chat_input(f"Ask me anything about {symbol}..."):
        # Add user message to chat history
        st.session_state.chat_messages[symbol].append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # Get stock context
        if symbol in st.session_state.stock_data:
            stock_context = st.session_state.stock_data[symbol]['context']

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_llm(prompt, stock_context)
                st.write(response)

            # Add AI response to chat history
            st.session_state.chat_messages[symbol].append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.write("Please analyze a stock first to get context for our conversation.")

    # Clear chat button
    if st.button(f"🗑️ Clear Chat", key=f"clear_chat_{symbol}"):
        st.session_state.chat_messages[symbol] = []
        st.rerun()


def process_stock_analysis(analyzer, symbol, period, chart_type, market):
    """Process stock analysis and store results"""
    with st.spinner(f"Fetching data for {symbol}..."):
        data, error = analyzer.fetch_stock_data(symbol, period)

        if error:
            st.error(error)
            return False

        if data is None or data.empty:
            st.error(f"No data available for {symbol}")
            return False

        # Calculate technical indicators
        data = analyzer.create_technical_indicators(data)
        currency_symbol = analyzer.market_currencies.get(market, '$')
        metrics = analyzer.calculate_metrics(data, currency_symbol)

        # Store in session state
        st.session_state.current_symbol = symbol
        st.session_state.stock_data[symbol] = {
            'data': data,
            'metrics': metrics,
            'context': get_stock_context_for_llm(symbol, metrics, data),
            'market': market,
            'period': period,
            'chart_type': chart_type
        }

        return True


def display_stock_analysis(analyzer, symbol):
    """Display stock analysis results"""
    if symbol not in st.session_state.stock_data:
        st.error("Stock data not found")
        return

    stock_info = st.session_state.stock_data[symbol]
    data = stock_info['data']
    metrics = stock_info['metrics']
    chart_type = stock_info['chart_type']

    # Display metrics
    st.subheader(f"📊 {symbol} Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", metrics.get('Current Price', 'N/A'))
    with col2:
        change_value = metrics.get('Period Change', 'N/A')
        change_pct = metrics.get('Period Change %', 'N/A')
        st.metric("Period Change", change_value, change_pct)
    with col3:
        st.metric("Period High", metrics.get('Period High', 'N/A'))
    with col4:
        st.metric("Period Low", metrics.get('Period Low', 'N/A'))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Volume", metrics.get('Average Volume', 'N/A'))
    with col2:
        st.metric("Volatility", metrics.get('Volatility', 'N/A'))
    with col3:
        st.metric("Current RSI", metrics.get('Current RSI', 'N/A'))
    with col4:
        current_rsi = data['RSI'].iloc[-1] if not data['RSI'].isna().iloc[-1] else None
        if current_rsi:
            if current_rsi > 70:
                signal = "Overbought"
            elif current_rsi < 30:
                signal = "Oversold"
            else:
                signal = "Neutral"
            st.metric("RSI Signal", signal)

    # Charts and Chat in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Chart", "🔧 Technical Analysis", "📊 RSI", "📈 Volume", "🤖 AI Chat"])

    with tab1:
        price_chart = analyzer.create_price_chart(data, symbol, chart_type)
        st.plotly_chart(price_chart, use_container_width=True)

    with tab2:
        technical_chart = analyzer.create_technical_chart(data, symbol)
        st.plotly_chart(technical_chart, use_container_width=True)

    with tab3:
        rsi_chart = analyzer.create_rsi_chart(data, symbol)
        st.plotly_chart(rsi_chart, use_container_width=True)

    with tab4:
        volume_chart = analyzer.create_volume_chart(data, symbol)
        st.plotly_chart(volume_chart, use_container_width=True)

    with tab5:
        display_chat_interface(symbol)

    # Additional options
    with st.expander("📋 Data & Export"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recent Data")
            st.dataframe(data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']], use_container_width=True)

        with col2:
            st.subheader("Analysis Summary")
            summary = analyzer.generate_ai_summary(symbol, metrics, data)
            st.write(summary)

            csv = data.to_csv()
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{symbol}_{stock_info['period']}_data.csv",
                mime="text/csv",
                use_container_width=True
            )


def main():
    initialize_session_state()

    st.title("📈 Stock Analysis Tool")
    st.markdown("*Analyze stocks from global markets with AI-powered insights*")

    analyzer = StockAnalyzer()

    # Main input section
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        query = st.text_input(
            "Enter stock symbol or query:",
            placeholder="e.g., 'AAPL 3 months' or 'RELIANCE 1 year'",
            help="Include stock symbol and optional time period"
        )

    with col2:
        market = st.selectbox(
            "Market",
            list(analyzer.market_suffixes.keys()),
            index=0
        )

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox("Chart Type", ["line", "candlestick"], index=0)
            manual_period = st.selectbox("Manual Period", ["1d", "1mo", "3mo", "6mo", "1y", "ytd"], index=2)
        with col2:
            manual_symbol = st.text_input("Manual Symbol", placeholder="e.g., AAPL")
            manual_analyze = st.button("🔍 Analyze Manual Input")

    # Process analysis
    if analyze_btn and query:
        symbol = analyzer.get_stock_symbol(query, market)
        timeframe = analyzer.get_timeframe(query)

        if not symbol:
            st.error("Could not extract stock symbol from query. Please include a valid stock symbol.")
        else:
            st.info(f"Analyzing: {symbol} | Timeframe: {timeframe}")
            if process_stock_analysis(analyzer, symbol, timeframe, chart_type, market):
                display_stock_analysis(analyzer, symbol)

    elif manual_analyze and manual_symbol:
        symbol = analyzer.format_symbol(manual_symbol, market)
        if process_stock_analysis(analyzer, symbol, manual_period, chart_type, market):
            display_stock_analysis(analyzer, symbol)

    # Display current analysis if available
    elif st.session_state.current_symbol and st.session_state.current_symbol in st.session_state.stock_data:
        display_stock_analysis(analyzer, st.session_state.current_symbol)


if __name__ == "__main__":
    main()