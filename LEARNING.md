# Learning Journal — QuantScope

A personal record of the technical concepts I encountered, struggled with, and understood while building this project.

---

## Streamlit

### Session state is the backbone of any stateful app

Early on I found that every user interaction in Streamlit triggers a full Python rerun from the top of the file. This was counterintuitive coming from a standard script mindset. The solution is `st.session_state` — a dict-like object that persists across reruns. I store the active stock symbol, fetched data, and chat message history in it so nothing is lost when the user clicks a button or types in a field.

### `@st.cache_data` and the `_self` trick

I used `@st.cache_data(ttl=300)` on the `fetch_stock_data` method to avoid hitting Yahoo Finance on every tiny interaction. The cache key is derived by hashing all function arguments. When the function is a method on a class, Streamlit tries to hash `self`, which fails with `UnhashableParamError` because class instances are not hashable by default. The fix is to name the parameter `_self` instead of `self` — the leading underscore tells Streamlit to skip hashing that argument. This was a non-obvious bug that took real debugging to track down.

### `@st.cache_resource` for heavyweight objects

LLM client objects (like `ChatGroq`) hold HTTP connection pools and should not be recreated on every rerun. `@st.cache_resource` keeps a single instance alive for the lifetime of the app. The cache key is the function arguments, so by passing `(provider, credential, model)` as a tuple I get one cached client per unique combination — switching provider or model automatically creates a new cached instance.

### Chat input and tab reset

`st.chat_input` submits a message by triggering a rerun. When the chat was inside an `st.tabs` container, every message submission reset the active tab back to the first one, because Streamlit rerenders tab state from scratch. I fixed this by moving the chat section outside the tabs entirely, rendered below a divider. The chat now persists across reruns without any tab state interference.

---

## Financial Data with yfinance

### Ticker symbols are market-specific

I learned that the same company can have different symbols depending on the exchange. Reliance Industries is `RELIANCE.NS` on the NSE, not just `RELIANCE`. yfinance uses a suffix convention to distinguish markets. I built a `MarketRegistry` that maps each supported market to its suffix so the app can automatically append the right one based on user selection, rather than expecting users to know the format themselves.

### Period strings

yfinance accepts period strings like `"3mo"`, `"1y"`, `"ytd"` directly in `stock.history(period=...)`. Learning these string formats and building a natural language parser to map phrases like "3 months" or "year to date" onto them made the query interface significantly more usable.

---

## Technical Indicators

### RSI from scratch

The Relative Strength Index is calculated over a rolling 14-period window. It measures the ratio of average gains to average losses. I implemented it using pandas `diff()`, `where()`, and `rolling().mean()`. The formula is `RSI = 100 - (100 / (1 + RS))` where `RS = average_gain / average_loss`. Values above 70 signal overbought and below 30 signal oversold — these thresholds are widely used in practice.

### Bollinger Bands

Bollinger Bands are built from a 20-period simple moving average (the middle band) plus and minus two standard deviations. They represent a dynamic price envelope. When price touches the upper band, it is statistically far above average; lower band implies the opposite. I implemented these directly with pandas `rolling().mean()` and `rolling().std()`.

### NaN propagation in rolling windows

A rolling window of size 20 means the first 19 rows will always be NaN since there are not enough preceding data points to calculate. I had to be careful when reading indicator values at specific rows — always checking `not data['RSI'].isna().iloc[-1]` before using the value to avoid passing NaN into strings or LLM context.

---

## LangChain

### The message format

LangChain uses a structured message format for chat models. A conversation is a list of message objects: `SystemMessage` sets the assistant's persona and context, while `HumanMessage` carries the user's actual question. I pass the full stock data context — price, RSI, SMAs, metrics — inside the `SystemMessage` so the LLM has everything it needs to give a grounded answer without hallucinating numbers.

### Provider abstraction

Each LLM provider (`ChatGroq`, `ChatOpenAI`, `ChatAnthropic`, `ChatOllama`) has the same `.invoke()` interface but slightly different constructor signatures. I centralised all provider instantiation in a single `_build_llm(provider, credential, model)` function that returns a `(llm, error)` tuple, so the rest of the app does not need to know which provider is active. This also made it straightforward to add new providers later — I only change one function.

### Import paths changed across langchain versions

Early versions of langchain had `langchain.schema.HumanMessage`. In langchain 1.x this was moved to `langchain_core.messages`. I ran into an `ImportError` at runtime after upgrading packages and learned that LangChain's major version increments come with significant module restructuring. Loose version pins (`>=`) across packages that pin each other tightly can result in conflicting transitive dependencies — I resolved this by switching to exact pins in `requirements.txt`.

---

## Dependency Management

### Why exact pins matter for deployment

On my local machine, `pip install -r requirements.txt` with `>=` bounds resolved to package versions that worked together. On Streamlit Cloud, the same file resolved to different (newer) versions that conflicted. The only reliable way to guarantee reproducible installs is to pin every direct dependency to an exact version. Using `pip freeze` to capture the known-good state of the local environment and committing that as `requirements.txt` is the correct workflow.

### Python version pinning with `runtime.txt`

Streamlit Cloud defaults to the latest available Python version. When Python 3.14 was the default, packages like `pandas==2.2.2` and `pillow` had no pre-built wheels for it, causing the build to attempt compiling from source — which failed due to missing system headers (`zlib`). A `runtime.txt` file containing `python-3.12` tells Streamlit Cloud exactly which Python to use and eliminates this class of problem entirely.

### Transitive dependencies

Some packages (`tiktoken`, `httpx`, `pydantic`) are not direct dependencies of my code but are required by langchain internals. On some environments they install automatically as transitive deps; on others they do not. Explicitly listing critical transitive deps in `requirements.txt` removes that ambiguity.

---

## Plotly

### Figure composition with traces

Plotly charts are built by adding traces to a `go.Figure()`. Each trace is a separate data series — I can add `go.Scatter` for lines, `go.Bar` for volume, and `go.Candlestick` for OHLC data to the same figure. `fig.update_layout()` controls titles, axes, height, and theme. The `template='plotly_white'` gives a clean white background consistent with Streamlit's default light theme.

### `use_container_width=True`

Passing `use_container_width=True` to `st.plotly_chart()` makes the chart fill the full width of whatever column or container it is rendered in, rather than using a fixed pixel width. This is essential for responsive layout across different screen sizes.

---

## Python Patterns

### Dataclasses for configuration

Using `@dataclass` for `MarketConfig` (fields: `name`, `suffix`, `currency`) gives me a typed, lightweight container without the boilerplate of a full class. It is cleaner than a nested dict and gives IDE autocomplete on field access.

### Registry pattern

`MarketRegistry` is a class with a single class-level dict (`MARKETS`) and two `@classmethod` methods. This is a simple registry pattern — a single place to define and look up configuration entries. I no longer have market suffixes hardcoded in multiple places across the codebase.

### Optional and the `or None` idiom

When reading from `st.session_state`, an empty string `""` is falsy but technically present. Using `return st.session_state.get(key) or None` converts empty strings to `None`, which I can then check with `if credential:` cleanly throughout the app.

---

## General Engineering

### Separate concerns between data fetching and rendering

I keep `process_stock_analysis()` (fetches data, calculates indicators, stores in session state) separate from `display_stock_analysis()` (reads from session state, renders UI). This means the analysis only runs when explicitly triggered by the user, while the display function can be called on every rerun to keep the analysis visible without re-fetching data.

### BYOK as a design constraint

Designing around BYOK (bring your own key) forced me to think about where credentials live, how long they persist, and what happens if they are absent. Keys are stored only in `st.session_state` for the duration of the browser session. They are never written to disk, never logged, and never sent anywhere except directly to the chosen provider SDK. The UI degrades gracefully when no key is present — the chat is disabled with a clear message instead of an unhandled exception.
