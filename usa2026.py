import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import time

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام تحليل الأسهم المتقدم",
    layout="wide",
    initial_sidebar_state="expanded"
)

# عنوان التطبيق
st.title('📈 نظام تحليل الأسهم الأمريكي المتقدم')
st.markdown("""
هذا التطبيق يقوم بما يلي:
- عرض الأخبار المالية العاجلة
- تتبع الأسهم الأكثر ارتفاعاً
- التحليل الفني المتقدم
- التنبؤ بحركة الأسهم باستخدام الذكاء الاصطناعي
""")

# API Keys (يجب استبدالها بمفاتيحك الخاصة)
NEWS_API_KEY = 'f55929edb5ee471791a1e622332ff6d8'  # احصل عليه من newsapi.org
#TIINGO_API_KEY = "16be092ddfdcb6e34f1de36875a3072e2c724afb"
# ---------------------------------------------------
# قسم الأخبار العاجلة
# ---------------------------------------------------
@st.cache_data(ttl=3600)  # تخزين النتائج لمدة ساعة
def get_financial_news():
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        news = newsapi.get_top_headlines(
            category='business',
            language='en',
            country='us'
        )
        return news['articles']
    except:
        return []

# ---------------------------------------------------
# قسم الأسهم الأكثر ارتفاعاً
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def get_top_gainers():
    try:
        url = "https://finance.yahoo.com/gainers"
        tables = pd.read_html(url)
        top_gainers = tables[0].head(10)
        return top_gainers
    except:
        return pd.DataFrame()

# ---------------------------------------------------
# وظائف التحليل الفني
# ---------------------------------------------------
def calculate_technical_indicators(data):
    # حساب مؤشر RSI
    rsi_indicator = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi_indicator.rsi()
    
    # حساب Bollinger Bands
    bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_high'] = bb_indicator.bollinger_hband()
    data['BB_low'] = bb_indicator.bollinger_lband()
    
    # حساب MACD
    macd_indicator = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd_indicator.macd()
    data['MACD_signal'] = macd_indicator.macd_signal()
    
    return data

def plot_technical_analysis(data, ticker):
    fig = go.Figure()
    
    # الرسم البياني للأسعار
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='أسعار الأسهم'
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_high'],
        line=dict(color='rgba(250, 0, 0, 0.3)'),
        name='النطاق العلوي'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_low'],
        line=dict(color='rgba(0, 250, 0, 0.3)'),
        name='النطاق السفلي'
    ))
    
    # إعدادات التخطيط
    fig.update_layout(
        title=f'التحليل الفني لسهم {ticker}',
        xaxis_title='التاريخ',
        yaxis_title='السعر',
        hovermode='x unified',
        height=600
    )
    
    return fig

# ---------------------------------------------------
# قسم التنبؤ بالأسهم
# ---------------------------------------------------
def prepare_data_for_prediction(data):
    # إنشاء متغيرات للتنبؤ
    data['Price_Up'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data = data.dropna()
    
    # تحديد الميزات والهدف
    features = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD']]
    target = data['Price_Up']
    
    return features, target

def train_prediction_model(features, target):
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # تدريب النموذج
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return model, mse

def predict_next_day(model, last_data):
    # تحضير بيانات اليوم الأخير للتنبؤ
    last_features = last_data[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD']].values.reshape(1, -1)
    prediction = model.predict(last_features)
    
    return prediction[0]

# ---------------------------------------------------
# واجهة المستخدم
# ---------------------------------------------------
def main():
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header('الإعدادات')
        ticker = st.text_input('أدخل رمز السهم (مثال: AAPL)', 'AAPL')
        start_date = st.date_input('تاريخ البداية', datetime.now() - timedelta(days=365))
        end_date = st.date_input('تاريخ النهاية', datetime.now())
        
        st.markdown("---")
        st.markdown("""
        **مصادر البيانات:**
        - Yahoo Finance
        - NewsAPI
        """)
    
    # تبويبات الواجهة
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 الصفحة الرئيسية",
        "📰 الأخبار العاجلة",
        "📊 التحليل الفني",
        "🔮 التنبؤ بالأسهم"
    ])
    
    with tab1:
        st.header("الأسهم الأكثر ارتفاعاً اليوم")
        gainers = get_top_gainers()
        if not gainers.empty:
            st.dataframe(gainers.style.highlight_max(axis=0))
        else:
            st.warning("لا يمكن جلب بيانات الأسهم الصاعدة حالياً. يرجى المحاولة لاحقاً.")
        
        st.markdown("---")
        st.header("أهم المؤشرات الأمريكية")
        
       # في قسم عرض مؤشرات السوق الرئيسية (تبويب الصفحة الرئيسية)
indices = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC"
}

cols = st.columns(3)
for i, (name, ticker_symbol) in enumerate(indices.items()):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    if not data.empty and 'Close' in data.columns:
        try:
            close_prices = data['Close']
            if len(close_prices) > 0:
                change = ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100
                cols[i].metric(
                    label=name,
                    value=f"{close_prices.iloc[-1]:.2f}",
                    delta=f"{change:.2f}%"
                )
            else:
                cols[i].metric(label=name, value="N/A", delta="N/A")
        except Exception as e:
            cols[i].metric(label=name, value="Error", delta=str(e))
    else:
        cols[i].metric(label=name, value="No Data", delta="N/A")
    
    with tab2:
        st.header("آخر الأخبار المالية العاجلة")
        news = get_financial_news()
        
        if news:
            for article in news[:10]:
                st.markdown(f"""
                ### {article['title']}
                **المصدر:** {article['source']['name']}  
                **التاريخ:** {article['publishedAt'][:10]}  
                {article['description']}  
                [قراءة المزيد]({article['url']})
                """)
                st.markdown("---")
        else:
            st.warning("لا يمكن جلب الأخبار حالياً. يرجى التحقق من اتصال الإنترنت أو مفتاح API.")
    
    with tab3:
        st.header("التحليل الفني المتقدم")
        
        if ticker:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty:
                # حساب المؤشرات الفنية
                data = calculate_technical_indicators(data)
                
                # عرض الرسم البياني
                st.plotly_chart(plot_technical_analysis(data, ticker), use_container_width=True)
                
                # عرض المؤشرات الفنية
                st.subheader("قراءات المؤشرات الفنية")
                cols = st.columns(4)
                
                # RSI
                current_rsi = data['RSI'][-1]
                rsi_status = "مشترى زائد (فوق 70)" if current_rsi > 70 else "مبيع زائد (تحت 30)" if current_rsi < 30 else "محايد"
                cols[0].metric("RSI (14 يوم)", f"{current_rsi:.2f}", rsi_status)
                
                # MACD
                macd_diff = data['MACD'][-1] - data['MACD_signal'][-1]
                macd_status = "إشارة شراء" if macd_diff > 0 else "إشارة بيع"
                cols[1].metric("MACD", f"{macd_diff:.2f}", macd_status)
                
                # Bollinger Bands
                current_price = data['Close'][-1]
                bb_high = data['BB_high'][-1]
                bb_low = data['BB_low'][-1]
                bb_status = "قرب المقاومة" if current_price > bb_high * 0.95 else "قرب الدعم" if current_price < bb_low * 1.05 else "في النطاق"
                cols[2].metric("Bollinger Bands", bb_status)
                
                # حجم التداول
                vol_change = ((data['Volume'][-1] - data['Volume'].mean()) / data['Volume'].mean()) * 100
                cols[3].metric("حجم التداول", f"{data['Volume'][-1]/1e6:.2f}M", f"{vol_change:.2f}%")
                
                # عرض بيانات الأسهم
                st.subheader("آخر 10 أيام تداول")
                st.dataframe(data.tail(10))
            else:
                st.error(f"لا توجد بيانات متاحة للسهم {ticker}. يرجى التحقق من الرمز.")
    
    with tab4:
        st.header("تنبؤ حركة السهم")
        
        if ticker:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if not data.empty and len(data) > 30:  # تحتاج إلى بيانات كافية
                # حساب المؤشرات الفنية
                data = calculate_technical_indicators(data)
                
                # تحضير البيانات للتنبؤ
                features, target = prepare_data_for_prediction(data)
                
                # تدريب النموذج
                with st.spinner('جاري تدريب نموذج التنبؤ...'):
                    model, mse = train_prediction_model(features, target)
                
                st.success(f"تم تدريب النموذج بنجاح (خطأ مربع متوسط: {mse:.4f})")
                
                # التنبؤ لليوم التالي
                last_data = data.iloc[-1]
                prediction = predict_next_day(model, last_data)
                
                # عرض النتائج
                st.subheader("توقعات الغد")
                cols = st.columns(2)
                
                cols[0].metric("السعر الحالي", f"{last_data['Close']:.2f}")
                
                if prediction > 0.6:
                    cols[1].metric("التوقع", "ارتفاع محتمل", delta="↑", delta_color="normal")
                elif prediction < 0.4:
                    cols[1].metric("التوقع", "انخفاض محتمل", delta="↓", delta_color="inverse")
                else:
                    cols[1].metric("التوقع", "مستقر", delta="→", delta_color="off")
                
                # تفسير النتائج
                st.markdown("""
                **كيفية تفسير التوقعات:**
                - **أعلى من 0.6:** إشارة شراء (احتمال ارتفاع السعر)
                - **أقل من 0.4:** إشارة بيع (احتمال انخفاض السعر)
                - **بين 0.4 و 0.6:** سوق متقلب/مستقر
                
                *ملاحظة: هذه التوقعات تعتمد على نماذج التعلم الآلي ولا تضمن الدقة الكاملة.*
                """)
                
                # أهمية الميزات
                st.subheader("أهم العوامل في التنبؤ")
                feature_importance = pd.DataFrame({
                    'الميزة': features.columns,
                    'الأهمية': model.feature_importances_
                }).sort_values('الأهمية', ascending=False)
                
                st.bar_chart(feature_importance.set_index('الميزة'))
            else:
                st.error("لا توجد بيانات كافية للتنبؤ. يحتاج النموذج إلى بيانات 30 يوم على الأقل.")

if __name__ == "__main__":
    main()
