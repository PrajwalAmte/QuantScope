"""
QuantScope: AI-Powered Stock Analysis Platform
Bring your own API key platform for multi-market stock analysis with technical indicators and AI insights.
"""

import logging
import re
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (for optional defaults, backward compatibility)
load_dotenv()


@dataclass
class MarketConfig:
    """Configuration for supported markets."""
    name: str
    suffix: str
    currency: str


class MarketRegistry:
    """Registry of supported markets with their configurations."""
    
    MARKETS: Dict[str, MarketConfig] = {
        'US': MarketConfig('US', '', '$'),
        'Indian': MarketConfig('Indian', '.NS', '₹'),
        'UK': MarketConfig('UK', '.L', '£'),
        'Canada': MarketConfig('Canada', '.TO', 'C$'),
        'Australia': MarketConfig('Australia', '.AX', 'A$'),
        'Germany': MarketConfig('Germany', '.DE', '€'),
        'France': MarketConfig('France', '.PA', '€'),
        'Japan': MarketConfig('Japan', '.T', '¥'),
        'Hong Kong': MarketConfig('Hong Kong', '.HK', 'HK$'),
    }

    @classmethod
    def get_all_markets(cls) -> List[str]:
        """Get list of all supported market names."""
        return list(cls.MARKETS.keys())

    @classmethod
    def get_market_config(cls, market: str) -> Optional[MarketConfig]:
        """Get configuration for a specific market."""
        return cls.MARKETS.get(market)


# Provider definitions: id -> (display name, env var, available flag, key url)
PROVIDERS: Dict[str, Dict] = {
    'groq':      {'label': 'Groq',      'env': 'GROQ_API_KEY',      'url': 'https://console.groq.com',
                  'models': ['meta-llama/llama-4-scout-17b-16e-instruct', 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it']},
    'openai':    {'label': 'OpenAI',    'env': 'OPENAI_API_KEY',    'url': 'https://platform.openai.com/api-keys',
                  'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']},
    'anthropic': {'label': 'Anthropic', 'env': 'ANTHROPIC_API_KEY', 'url': 'https://console.anthropic.com/settings/keys',
                  'models': ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-3-5-haiku-20241022']},
    'ollama':    {'label': 'Ollama (local)', 'env': None,           'url': 'https://ollama.com/download',
                  'models': ['llama3.2', 'llama3.1', 'mistral', 'gemma2', 'qwen2.5']},
}


class APIKeyManager:
    """Manages API key and provider selection from Streamlit session state."""

    PROVIDER_SESSION     = 'llm_provider'
    MODEL_SESSION        = 'llm_model'
    GROQ_KEY_SESSION     = 'groq_api_key'
    OPENAI_KEY_SESSION   = 'openai_api_key'
    ANTHROPIC_KEY_SESSION = 'anthropic_api_key'
    OLLAMA_URL_SESSION   = 'ollama_base_url'
    PERSIST_KEYS_SESSION = 'persist_api_keys'

    @staticmethod
    def initialize_session() -> None:
        """Initialize session state for API key management."""
        defaults = {
            APIKeyManager.PROVIDER_SESSION:      'groq',
            APIKeyManager.MODEL_SESSION:         PROVIDERS['groq']['models'][0],
            APIKeyManager.GROQ_KEY_SESSION:      os.getenv('GROQ_API_KEY', ''),
            APIKeyManager.OPENAI_KEY_SESSION:    os.getenv('OPENAI_API_KEY', ''),
            APIKeyManager.ANTHROPIC_KEY_SESSION: os.getenv('ANTHROPIC_API_KEY', ''),
            APIKeyManager.OLLAMA_URL_SESSION:    'http://localhost:11434',
            APIKeyManager.PERSIST_KEYS_SESSION:  False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get_active_provider() -> str:
        return st.session_state.get(APIKeyManager.PROVIDER_SESSION, 'groq')

    @staticmethod
    def get_active_model() -> str:
        return st.session_state.get(APIKeyManager.MODEL_SESSION, PROVIDERS['groq']['models'][0])

    @staticmethod
    def get_key_for_provider(provider: str) -> Optional[str]:
        """Return the API key (or base URL for Ollama) for any provider."""
        mapping = {
            'groq':      APIKeyManager.GROQ_KEY_SESSION,
            'openai':    APIKeyManager.OPENAI_KEY_SESSION,
            'anthropic': APIKeyManager.ANTHROPIC_KEY_SESSION,
            'ollama':    APIKeyManager.OLLAMA_URL_SESSION,
        }
        return st.session_state.get(mapping.get(provider, '')) or None

    # Legacy helper kept for backward compatibility
    @staticmethod
    def get_groq_key() -> Optional[str]:
        return APIKeyManager.get_key_for_provider('groq')

    @staticmethod
    def set_groq_key(key: str) -> None:
        st.session_state[APIKeyManager.GROQ_KEY_SESSION] = key
        logger.info("Groq API key updated in session")


def display_api_key_sidebar() -> None:
    """Display LLM provider selection and API key inputs in the sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("AI Provider")
        st.caption("QuantScope is bring-your-own-key. Stock data is always free.")

        # Provider selector
        provider_labels = [PROVIDERS[p]['label'] for p in PROVIDERS]
        provider_ids    = list(PROVIDERS.keys())
        current_provider = st.session_state.get(APIKeyManager.PROVIDER_SESSION, 'groq')
        provider_idx = provider_ids.index(current_provider) if current_provider in provider_ids else 0

        selected_label = st.selectbox(
            "Provider",
            provider_labels,
            index=provider_idx,
            key="provider_selectbox"
        )
        selected_provider = provider_ids[provider_labels.index(selected_label)]
        if selected_provider != current_provider:
            st.session_state[APIKeyManager.PROVIDER_SESSION] = selected_provider
            st.session_state[APIKeyManager.MODEL_SESSION] = PROVIDERS[selected_provider]['models'][0]
            st.rerun()

        provider_cfg = PROVIDERS[selected_provider]

        # Model selector
        current_model = st.session_state.get(APIKeyManager.MODEL_SESSION, provider_cfg['models'][0])
        if current_model not in provider_cfg['models']:
            current_model = provider_cfg['models'][0]
        model_idx = provider_cfg['models'].index(current_model)
        selected_model = st.selectbox("Model", provider_cfg['models'], index=model_idx, key="model_selectbox")
        st.session_state[APIKeyManager.MODEL_SESSION] = selected_model

        # Credential input
        if selected_provider == 'ollama':
            current_url = st.session_state.get(APIKeyManager.OLLAMA_URL_SESSION, 'http://localhost:11434')
            url_input = st.text_input(
                "Ollama Base URL",
                value=current_url,
                help="Default: http://localhost:11434",
                key="ollama_url_input"
            )
            if url_input != current_url:
                st.session_state[APIKeyManager.OLLAMA_URL_SESSION] = url_input
        else:
            session_key_map = {
                'groq':      APIKeyManager.GROQ_KEY_SESSION,
                'openai':    APIKeyManager.OPENAI_KEY_SESSION,
                'anthropic': APIKeyManager.ANTHROPIC_KEY_SESSION,
            }
            sess_key = session_key_map[selected_provider]
            current_key = st.session_state.get(sess_key, '')
            key_input = st.text_input(
                f"{provider_cfg['label']} API Key",
                value=current_key,
                type="password",
                help=f"Get your key at {provider_cfg['url']}",
                key=f"{selected_provider}_key_input"
            )
            if key_input != current_key:
                st.session_state[sess_key] = key_input

        # Test Connection
        credential = APIKeyManager.get_key_for_provider(selected_provider)
        test_key = f"conn_test_{selected_provider}"

        if not credential:
            st.caption("Enter credentials above to test the connection.")
        else:
            if st.button("Test Connection", key="test_conn_btn", use_container_width=True):
                with st.spinner("Testing connection..."):
                    llm, err = _build_llm(selected_provider, credential, selected_model)
                    if err:
                        st.session_state[test_key] = ('error', err)
                    else:
                        try:
                            llm.invoke([HumanMessage(content="Reply with OK only.")])
                            st.session_state[test_key] = ('ok', f"Connected to {provider_cfg['label']} successfully.")
                        except Exception as e:
                            st.session_state[test_key] = ('error', str(e))

            if test_key in st.session_state:
                status, msg = st.session_state[test_key]
                if status == 'ok':
                    st.success(msg)
                else:
                    st.error(msg)

        st.divider()


@st.cache_resource
def _build_llm(provider: str, credential: str, model: str):
    """Build and cache an LLM instance for the given provider/model/credential tuple."""
    try:
        if provider == 'groq':
            return ChatGroq(api_key=credential, model=model, temperature=0.7, max_tokens=1500), None

        if provider == 'openai':
            if not OPENAI_AVAILABLE:
                return None, "langchain-openai is not installed. Run: pip install langchain-openai"
            return ChatOpenAI(api_key=credential, model=model, temperature=0.7, max_tokens=1500), None

        if provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                return None, "langchain-anthropic is not installed. Run: pip install langchain-anthropic"
            return ChatAnthropic(api_key=credential, model=model, temperature=0.7, max_tokens=1500), None

        if provider == 'ollama':
            if not OLLAMA_AVAILABLE:
                return None, "langchain-ollama is not installed. Run: pip install langchain-ollama"
            return ChatOllama(base_url=credential, model=model), None

        return None, f"Unknown provider: {provider}"
    except Exception as e:
        error_msg = f"Failed to initialize {provider} LLM: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


# Keep backward-compatible name used in older call sites
def initialize_groq_llm(api_key: str) -> tuple:
    return _build_llm('groq', api_key, PROVIDERS['groq']['models'][0])


def initialize_session_state() -> None:
    """Initialize all session state variables."""
    default_session_vars = {
        'initialized': True,
        'current_symbol': None,
        'stock_data': {},
        'chat_messages': {},
    }
    
    for key, value in default_session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    APIKeyManager.initialize_session()


def get_stock_context_for_llm(symbol: str, metrics: Dict[str, str], data: pd.DataFrame) -> str:
    """Prepare stock context for LLM analysis."""
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


def chat_with_llm(user_question: str, stock_context: str, api_key: str) -> str:
    """Chat with the active LLM provider about the stock."""
    provider  = APIKeyManager.get_active_provider()
    model     = APIKeyManager.get_active_model()
    credential = APIKeyManager.get_key_for_provider(provider)

    llm, error = _build_llm(provider, credential or '', model)

    if error:
        logger.error(f"LLM initialization failed: {error}")
        return f"AI Chat unavailable: {error}"

    if not llm:
        return "LLM is not available. Please configure a provider in the sidebar."

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
        logger.info(f"LLM response generated for question: {user_question[:50]}...")
        return response.content

    except Exception as e:
        error_msg = f"Error communicating with LLM: {str(e)}"
        logger.error(error_msg)
        return error_msg


class StockAnalyzer:
    """Core stock analysis engine with data fetching, calculations, and visualizations."""
    
    TIMEFRAME_MAPPING: Dict[str, List[str]] = {
        'ytd': ['ytd', 'year to date'],
        '1y': ['1y', 'year', 'annual'],
        '6mo': ['6mo', '6 month', 'half year'],
        '3mo': ['3mo', '3 month', 'quarter'],
        '1mo': ['1mo', 'month'],
        '1d': ['1d', 'day', 'today']
    }
    
    COMMON_WORDS = {'FOR', 'AND', 'THE', 'OF', 'TO', 'IN', 'ON', 'AT', 'BY', 'WITH'}

    def format_symbol(self, symbol: str, market: str) -> str:
        """Format stock symbol based on market suffix."""
        market_config = MarketRegistry.get_market_config(market)
        if not market_config:
            return symbol.upper().strip()
        
        symbol = symbol.upper().strip()
        suffix = market_config.suffix

        # Remove existing suffixes
        for suf in [cfg.suffix for cfg in MarketRegistry.MARKETS.values()]:
            if symbol.endswith(suf):
                symbol = symbol.replace(suf, '')
                break

        return f"{symbol}{suffix}"

    def get_stock_symbol(self, query: str, market: str) -> Optional[str]:
        """Extract stock symbol from natural language query."""
        words = query.upper().split()

        for word in words:
            if re.match(r'^[A-Z0-9]{2,10}$', word) and word not in self.COMMON_WORDS:
                return self.format_symbol(word, market)
        return None

    def get_timeframe(self, query: str) -> str:
        """Extract timeframe from natural language query."""
        query_lower = query.lower()

        for period, keywords in self.TIMEFRAME_MAPPING.items():
            if any(keyword in query_lower for keyword in keywords):
                return period

        return '3mo'  # Default timeframe

    @st.cache_data(ttl=300)
    def fetch_stock_data(_self, symbol: str, period: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Fetch stock data using yfinance with caching."""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                error_msg = f"No data found for symbol: {symbol}"
                logger.warning(error_msg)
                return None, error_msg
            
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data, None
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def create_price_chart(self, data: pd.DataFrame, symbol: str, chart_type: str = "line") -> go.Figure:
        """Create interactive price chart."""
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

    def create_volume_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create volume chart."""
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

    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators: SMA, RSI, Bollinger Bands."""
        data = data.copy()

        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()

        # RSI calculation (14-period)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-period, 2 std dev)
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

        return data

    def create_technical_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create technical indicators visualization."""
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

    def create_rsi_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create RSI indicator chart."""
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

    def calculate_metrics(self, data: pd.DataFrame, currency_symbol: str) -> Dict[str, str]:
        """Calculate key financial metrics from stock data."""
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

    def generate_summary(self, symbol: str, metrics: Dict[str, str], data: pd.DataFrame) -> str:
        """Generate analysis summary based on stock data and metrics."""
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
        except ValueError:
            trend = "neutral"

        volatility = metrics.get("Volatility", "N/A")
        try:
            vol_val = float(volatility.replace('%', ''))
            vol_level = "high" if vol_val > 2 else "low"
        except ValueError:
            vol_level = "moderate"

        return (
            f"The current price of **{symbol}** is {current}, with a net change of {change} "
            f"({change_pct}) over the selected period. The RSI value is {rsi}, suggesting a **{trend}** trend. "
            f"Volatility is {volatility}, indicating a **{vol_level}** fluctuation in price."
        )


def display_chat_interface(symbol: str) -> None:
    """Display chat interface for stock analysis questions."""
    provider   = APIKeyManager.get_active_provider()
    credential = APIKeyManager.get_key_for_provider(provider)
    provider_label = PROVIDERS.get(provider, {}).get('label', provider)

    st.subheader(f"Chat about {symbol}")
    st.caption(f"Provider: {provider_label} · Model: {APIKeyManager.get_active_model()}")

    if not credential:
        st.warning(
            f"No credential configured for {provider_label}. "
            "Please enter your API key in the sidebar to use AI Chat."
        )
        provider_url = PROVIDERS.get(provider, {}).get('url', '')
        if provider_url:
            st.info(f"Get your key at {provider_url}")
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

    # Chat input
    if prompt := st.chat_input(f"Ask me anything about {symbol}..."):
        st.session_state.chat_messages[symbol].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Get stock context
        if symbol in st.session_state.stock_data:
            stock_context = st.session_state.stock_data[symbol]['context']

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_llm(prompt, stock_context, credential)
                st.write(response)

            st.session_state.chat_messages[symbol].append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.write("Please analyze a stock first to get context for our conversation.")

    # Clear chat button
    if st.button("Clear Chat", key=f"clear_chat_{symbol}"):
        st.session_state.chat_messages[symbol] = []
        st.rerun()


def process_stock_analysis(
    analyzer: StockAnalyzer,
    symbol: str,
    period: str,
    chart_type: str,
    market: str
) -> bool:
    """Process stock analysis and store results in session state."""
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
        
        market_config = MarketRegistry.get_market_config(market)
        currency_symbol = market_config.currency if market_config else '$'
        
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

        logger.info(f"Analysis processed for {symbol} ({period})")
        return True


def display_stock_analysis(analyzer: StockAnalyzer, symbol: str) -> None:
    """Display complete stock analysis with charts and metrics."""
    if symbol not in st.session_state.stock_data:
        st.error("Stock data not found")
        return

    stock_info = st.session_state.stock_data[symbol]
    data = stock_info['data']
    metrics = stock_info['metrics']
    chart_type = stock_info['chart_type']

    # Display metrics
    st.subheader(f"{symbol} Overview")

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
        current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and not data['RSI'].isna().iloc[-1] else None
        if current_rsi:
            if current_rsi > 70:
                signal = "Overbought"
            elif current_rsi < 30:
                signal = "Oversold"
            else:
                signal = "Neutral"
            st.metric("RSI Signal", signal)

    # Charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Analysis", "RSI", "Volume"])

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

    # AI Chat — rendered outside tabs so it persists across reruns
    st.divider()
    display_chat_interface(symbol)

    # Data & Export
    with st.expander("Data & Export"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recent Data")
            st.dataframe(data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']], use_container_width=True)

        with col2:
            st.subheader("Analysis Summary")
            summary = analyzer.generate_summary(symbol, metrics, data)
            st.write(summary)

            csv = data.to_csv()
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"{symbol}_{stock_info['period']}_data.csv",
                mime="text/csv",
                use_container_width=True
            )


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="QuantScope - Stock Analysis",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize state
    initialize_session_state()
    display_api_key_sidebar()

    st.title("QuantScope")
    st.markdown("*Bring-Your-Own-Key stock analysis with AI insights across global markets*")

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
            MarketRegistry.get_all_markets(),
            index=0
        )

    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox("Chart Type", ["line", "candlestick"], index=0)
            manual_period = st.selectbox("Manual Period", ["1d", "1mo", "3mo", "6mo", "1y", "ytd"], index=2)
        with col2:
            manual_symbol = st.text_input("Manual Symbol", placeholder="e.g., AAPL")
            manual_analyze = st.button("Analyze Manual Input")

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
