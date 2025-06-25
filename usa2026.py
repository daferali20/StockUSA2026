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

# ---------------------------------------------------
# قسم الأخبار العاجلة
# ---------------------------------------------------
@st.cache_data(ttl=3600)  # تخزين النتائج لمدة ساعة
def get_financial_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        news = newsapi.get_top_headlines(
            category='business',
            language='en',
            country='us'
        )
        return news.get('articles', [])
    except Exception as e:
        st.error(f"حدث خطأ في جلب الأخبار: {str(e)}")
        return []

# ---------------------------------------------------
# قسم الأسهم الأكثر ارتفاعاً
# ---------------------------------------------------
@st.cache_data(ttl=3600)
def get_top_gainers():
    try:
        url = "https://finance.yahoo.com/gainers"
        tables = pd.read_html(url)
        return tables[0].head(10) if tables else pd.DataFrame()
    except Exception as e:
        st.error(f"حدث خطأ في جلب الأسهم الصاعدة: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------
# وظائف التحليل الفني
# ---------------------------------------------------
def calculate_technical_indicators(data):
    try:
        if data.empty or 'Close' not in data.columns:
            return data
            
        # حساب مؤشر RSI
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        
        # حساب Bollinger Bands
        bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
        data['BB_high'] = bb_indicator.bollinger_hband()
        data['BB_low'] = bb_indicator.bollinger_lband()
        
        # حساب MACD
        macd_indicator = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd_indicator.macd()
        data['MACD_signal'] = macd_indicator.macd_signal()
        
        return data
    except Exception as e:
        st.error(f"حدث خطأ في حساب المؤشرات الفنية: {str(e)}")
        return data

def plot_technical_analysis(data, ticker):
    fig = go.Figure()
    
    try:
        # التحقق من وجود الأعمدة المطلوبة
        required_cols = ['Open', 'High', 'Low', 'Close', 'BB_high', 'BB_low']
        if not all(col in data.columns for col in required_cols):
            st.error("بيانات الأسعار غير مكتملة للتحليل الفني")
            return fig
            
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
    except Exception as e:
        st.error(f"حدث خطأ في إنشاء الرسم البياني: {str(e)}")
    
    return fig

# ---------------------------------------------------
# قسم التنبؤ بالأسهم
# ---------------------------------------------------
def prepare_data_for_prediction(data):
    try:
        if data.empty or 'Close' not in data.columns:
            return pd.DataFrame(), pd.Series()
            
        # إنشاء متغيرات للتنبؤ
        data['Price_Up'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        data = data.dropna()
        
        # تحديد الميزات والهدف
        features = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD']]
        target = data['Price_Up']
        
        return features, target
    except Exception as e:
        st.error(f"حدث خطأ في تحضير بيانات التنبؤ: {str(e)}")
        return pd.DataFrame(), pd.Series()

def train_prediction_model(features, target):
    try:
        if features.empty or target.empty:
            return None, 0
            
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
    except Exception as e:
        st.error(f"حدث خطأ في تدريب النموذج: {str(e)}")
        return None, 0

def predict_next_day(model, last_data):
    try:
        if model is None or last_data.empty:
            return 0.5
            
        required_cols = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD']
        if not all(col in last_data.index for col in required_cols):
            return 0.5
            
        # تحضير بيانات اليوم الأخير للتنبؤ
        last_features = last_data[required_cols].values.reshape(1, -1)
        return model.predict(last_features)[0]
    except:
        return 0.5

# ---------------------------------------------------
# واجهة المستخدم
# ---------------------------------------------------
def main():
    # شريط جانبي للإعدادات
    with st.sidebar:
        st.header('الإعدادات')
        ticker = st.text_input('أدخل رمز السهم (مثال: AAPL)', 'AAPL').strip().upper()
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
        
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "Nasdaq": "^IXIC"
        }

        cols = st.columns(3)
        for i, (name, ticker_symbol) in enumerate(indices.items()):
            try:
                data = yf.download(ticker_symbol, start=start_date, end=end_date)
                if not data.empty and 'Close' in data.columns and len(data['Close']) > 0:
                    close_prices = data['Close']
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
    
    with tab2:
        st.header("آخر الأخبار المالية العاجلة")
        news = get_financial_news()
        
        if news:
            for article in news[:10]:
                st.markdown(f"""
                ### {article.get('title', 'لا يوجد عنوان')}
                **المصدر:** {article.get('source', {}).get('name', 'غير معروف')}  
                **التاريخ:** {article.get('publishedAt', '')[:10]}  
                {article.get('description', 'لا يوجد وصف')}  
                [قراءة المزيد]({article.get('url', '#')})
                """)
                st.markdown("---")
        else:
            st.warning("لا يمكن جلب الأخبار حالياً. يرجى التحقق من اتصال الإنترنت أو مفتاح API.")
    
    with tab3:
        st.header("التحليل الفني المتقدم")
        
        if ticker:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if not data.empty:
                    # حساب المؤشرات الفنية
                    data = calculate_technical_indicators(data)
                    
                    # عرض الرسم البياني
                    fig = plot_technical_analysis(data, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # عرض المؤشرات الفنية
                    st.subheader("قراءات المؤشرات الفنية")
                    cols = st.columns(4)
                    
                    # RSI
                    if 'RSI' in data.columns:
                        current_rsi = data['RSI'].iloc[-1]
                        rsi_status = "مشترى زائد (فوق 70)" if current_rsi > 70 else "مبيع زائد (تحت 30)" if current_rsi < 30 else "محايد"
                        cols[0].metric("RSI (14 يوم)", f"{current_rsi:.2f}", rsi_status)
                    
                    # MACD
                    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                        macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
                        macd_status = "إشارة شراء" if macd_diff > 0 else "إشارة بيع"
                        cols[1].metric("MACD", f"{macd_diff:.2f}", macd_status)
                    
                    # Bollinger Bands
                    if all(col in data.columns for col in ['Close', 'BB_high', 'BB_low']):
                        current_price = data['Close'].iloc[-1]
                        bb_high = data['BB_high'].iloc[-1]
                        bb_low = data['BB_low'].iloc[-1]
                        bb_status = "قرب المقاومة" if current_price > bb_high * 0.95 else "قرب الدعم" if current_price < bb_low * 1.05 else "في النطاق"
                        cols[2].metric("Bollinger Bands", bb_status)
                    
                    # حجم التداول
                    if 'Volume' in data.columns:
                        vol_mean = data['Volume'].mean()
                        if vol_mean > 0:
                            vol_change = ((data['Volume'].iloc[-1] - vol_mean) / vol_mean) * 100
                            cols[3].metric("حجم التداول", f"{data['Volume'].iloc[-1]/1e6:.2f}M", f"{vol_change:.2f}%")
                    
                    # عرض بيانات الأسهم
                    st.subheader("آخر 10 أيام تداول")
                    st.dataframe(data.tail(10))
                else:
                    st.error(f"لا توجد بيانات متاحة للسهم {ticker}. يرجى التحقق من الرمز.")
            except Exception as e:
                st.error(f"حدث خطأ في تحميل بيانات السهم: {str(e)}")
    
    with tab4:
        st.header("تنبؤ حركة السهم")
        
        if ticker:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if not data.empty and len(data) > 30:
                    # حساب المؤشرات الفنية
                    data = calculate_technical_indicators(data)
                    
                    # تحضير البيانات للتنبؤ
                    features, target = prepare_data_for_prediction(data)
                    
                    if not features.empty and not target.empty:
                        # تدريب النموذج
                        with st.spinner('جاري تدريب نموذج التنبؤ...'):
                            model, mse = train_prediction_model(features, target)
                        
                        if model:
                            st.success(f"تم تدريب النموذج بنجاح (خطأ مربع متوسط: {mse:.4f})")
                            
                            # التنبؤ لليوم التالي
                            last_data = data.iloc[-1]
                            prediction = predict_next_day(model, last_data)
                            
                            # عرض النتائج
                            st.subheader("توقعات الغد")
                            cols = st.columns(2)
                            
                            cols[0].metric("السعر الحالي", f"{last_data.get('Close', 0):.2f}")
                            
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
                        st.error("لا توجد بيانات كافية للتنبؤ.")
                else:
                    st.error("لا توجد بيانات كافية للتنبؤ. يحتاج النموذج إلى بيانات 30 يوم على الأقل.")
            except Exception as e:
                st.error(f"حدث خطأ في عملية التنبؤ: {str(e)}")

if __name__ == "__main__":
    main()
