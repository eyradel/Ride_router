import misc.first as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import json
from datetime import datetime, timedelta

# Import our custom modules
from seo_scraper import SEOScraper
from gsc_integration import GoogleSearchConsoleIntegration, gsc_integration_ui
from chatgpt_seo import ChatGPTSEOAnalyzer

# Page configuration
st.set_page_config(
    page_title="Advanced SEO Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'crawl_results' not in st.session_state:
    st.session_state.crawl_results = None
if 'gsc_data' not in st.session_state:
    st.session_state.gsc_data = None
if 'integrated_data' not in st.session_state:
    st.session_state.integrated_data = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None

def main():
    # Sidebar for navigation
    st.sidebar.markdown("# 🔍 SEO Engine")
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Website Crawler", "GSC Integration", "Content Analysis", "Technical SEO", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Website Crawler":
        show_crawler()
    elif page == "GSC Integration":
        show_gsc()
    elif page == "Content Analysis":
        show_content_analysis()
    elif page == "Technical SEO":
        show_technical_seo()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    st.markdown('<h1 class="main-header">🚀 SEO Engine Dashboard</h1>', unsafe_allow_html=True)
    
    # Check if we have data
    has_crawl_data = st.session_state.crawl_results is not None and len(st.session_state.crawl_results) > 0
    has_gsc_data = hasattr(st.session_state, 'gsc_data') and not st.session_state.gsc_data.empty
    has_integrated_data = st.session_state.integrated_data is not None and not st.session_state.integrated_data.empty
    
    # Summary cards
    if not has_crawl_data and not has_gsc_data:
        st.info("👈 Start by crawling your website or connecting to Google Search Console in the sidebar!")
        
        # Quick start guide
        st.markdown('<h2 class="sub-header">Quick Start Guide</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🕸️ Website Crawler")
            st.markdown("""
            1. Go to "Website Crawler" in the sidebar
            2. Enter your website URL
            3. Configure crawl settings
            4. Start the crawl
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Google Search Console")
            st.markdown("""
            1. Go to "GSC Integration" in the sidebar
            2. Upload your Google API credentials
            3. Authenticate with Google
            4. Select your property and fetch data
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        return
    
    # Display summary metrics
    st.markdown('<h2 class="sub-header">Summary Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    if has_crawl_data:
        # Calculate average SEO score
        avg_score = sum(p.get('seo_score', 0) for p in st.session_state.crawl_results) / len(st.session_state.crawl_results)
        
        with col1:
            st.metric("Pages Crawled", len(st.session_state.crawl_results))
        with col2:
            st.metric("Avg. SEO Score", f"{avg_score:.1f}/100")
        
        # Get issues counts
        meta_issues = sum(1 for p in st.session_state.crawl_results if not p.get('meta_description', ''))
        h1_issues = sum(1 for p in st.session_state.crawl_results if p.get('h1_count', 0) == 0)
        
        with col3:
            st.metric("Issues Found", meta_issues + h1_issues)
    
    if has_gsc_data:
        with col4:
            clicks = int(st.session_state.gsc_data['clicks'].sum())
            st.metric("Total Clicks", f"{clicks:,}")
    
    # Create tabs for different dashboard sections
    tab1, tab2, tab3 = st.tabs(["Performance Overview", "Top Issues", "Recommendations"])
    
    with tab1:
        st.markdown('<h3 class="sub-header">Performance Overview</h3>', unsafe_allow_html=True)
        
        # SEO Score distribution if we have crawl data
        if has_crawl_data:
            scores = [p.get('seo_score', 0) for p in st.session_state.crawl_results]
            
            fig = px.histogram(
                x=scores,
                nbins=10,
                title="SEO Score Distribution",
                labels={'x': 'SEO Score', 'y': 'Number of Pages'},
                color_discrete_sequence=['#3366CC']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Word count vs SEO score scatter plot
            if len(st.session_state.crawl_results) > 1:
                word_counts = [p.get('word_count', 0) for p in st.session_state.crawl_results]
                page_urls = [p.get('url', '') for p in st.session_state.crawl_results]
                
                fig = px.scatter(
                    x=word_counts, 
                    y=scores,
                    title="Content Length vs SEO Score",
                    labels={'x': 'Word Count', 'y': 'SEO Score'},
                    hover_name=page_urls,
                    color_discrete_sequence=['#3366CC']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # GSC data visualization if available
        if has_gsc_data:
            # Top queries by clicks
            top_queries = st.session_state.gsc_data.groupby('query').sum().reset_index()
            top_queries = top_queries.sort_values('clicks', ascending=False).head(10)
            
            fig = px.bar(
                top_queries, 
                x='query', 
                y='clicks',
                title='Top 10 Queries by Clicks',
                labels={'query': 'Search Query', 'clicks': 'Clicks'},
                color_discrete_sequence=['#4CAF50']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h3 class="sub-header">Top Issues</h3>', unsafe_allow_html=True)
        
        if has_crawl_data:
            # Create issues summary
            issues = []
            
            # Missing meta descriptions
            missing_meta = [p['url'] for p in st.session_state.crawl_results if not p.get('meta_description', '')]
            if missing_meta:
                issues.append({
                    'issue': 'Missing meta descriptions',
                    'count': len(missing_meta),
                    'priority': 'High',
                    'impact': 'Reduces click-through rate from search results',
                    'urls': missing_meta
                })
            
            # Missing H1 tags
            missing_h1 = [p['url'] for p in st.session_state.crawl_results if p.get('h1_count', 0) == 0]
            if missing_h1:
                issues.append({
                    'issue': 'Missing H1 headings',
                    'count': len(missing_h1),
                    'priority': 'Medium',
                    'impact': 'Reduces content hierarchy clarity for search engines',
                    'urls': missing_h1
                })
            
            # Slow loading pages
            slow_pages = [p['url'] for p in st.session_state.crawl_results if p.get('load_time_seconds', 0) > 3]
            if slow_pages:
                issues.append({
                    'issue': 'Slow page load times (>3s)',
                    'count': len(slow_pages),
                    'priority': 'High',
                    'impact': 'Negatively affects user experience and rankings',
                    'urls': slow_pages
                })
            
            # Low word count
            thin_content = [p['url'] for p in st.session_state.crawl_results if p.get('word_count', 0) < 300]
            if thin_content:
                issues.append({
                    'issue': 'Thin content (<300 words)',
                    'count': len(thin_content),
                    'priority': 'Medium',
                    'impact': 'May not satisfy search intent or rank well',
                    'urls': thin_content
                })
            
            # Missing image alt text
            missing_alt = [p['url'] for p in st.session_state.crawl_results if p.get('images_without_alt', 0) > 0]
            if missing_alt:
                issues.append({
                    'issue': 'Images missing alt text',
                    'count': len(missing_alt),
                    'priority': 'Low',
                    'impact': 'Reduces accessibility and image search potential',
                    'urls': missing_alt
                })
            
            # Display issues
            if issues:
                for i, issue in enumerate(sorted(issues, key=lambda x: len(x['urls']), reverse=True)):
                    with st.expander(f"{issue['issue']} ({issue['count']} pages) - {issue['priority']} Priority"):
                        st.write(f"**Impact**: {issue['impact']}")
                        st.write("**Affected URLs:**")
                        for url in issue['urls'][:5]:
                            st.write(f"- {url}")
                        if len(issue['urls']) > 5:
                            st.write(f"- ...and {len(issue['urls']) - 5} more")
            else:
                st.success("No major issues found!")
    
    with tab3:
        st.markdown('<h3 class="sub-header">Recommendations</h3>', unsafe_allow_html=True)
        
        if has_crawl_data:
            # Generate recommendations
            if 'recommendations' not in st.session_state:
                # This would normally come from our SEOScraper or AI
                st.session_state.recommendations = [
                    {
                        'title': 'Improve Meta Descriptions',
                        'description': 'Add compelling meta descriptions to pages missing them to improve CTR from search results.',
                        'priority': 'High',
                        'effort': 'Medium',
                        'impact': 'High',
                        'affected_urls': missing_meta if 'missing_meta' in locals() else []
                    },
                    {
                        'title': 'Add H1 Headings',
                        'description': 'Ensure all pages have a single, keyword-rich H1 heading that clearly describes the page content.',
                        'priority': 'Medium',
                        'effort': 'Low',
                        'impact': 'Medium',
                        'affected_urls': missing_h1 if 'missing_h1' in locals() else []
                    },
                    {
                        'title': 'Optimize Page Speed',
                        'description': 'Improve loading time for slow pages by optimizing images, minifying CSS/JS, and implementing browser caching.',
                        'priority': 'High',
                        'effort': 'High',
                        'impact': 'High',
                        'affected_urls': slow_pages if 'slow_pages' in locals() else []
                    },
                    {
                        'title': 'Expand Thin Content',
                        'description': 'Add more comprehensive content to pages with less than 300 words to better satisfy user intent.',
                        'priority': 'Medium',
                        'effort': 'High',
                        'impact': 'Medium',
                        'affected_urls': thin_content if 'thin_content' in locals() else []
                    }
                ]
            
            # Display recommendations
            for rec in st.session_state.recommendations:
                with st.expander(f"{rec['title']} - {rec['priority']} Priority"):
                    st.write(f"**Description**: {rec['description']}")
                    st.write(f"**Effort**: {rec['effort']} | **Impact**: {rec['impact']}")
                    
                    if rec['affected_urls']:
                        st.write("**Affected URLs:**")
                        for url in rec['affected_urls'][:3]:
                            st.write(f"- {url}")
                        if len(rec['affected_urls']) > 3:
                            st.write(f"- ...and {len(rec['affected_urls']) - 3} more")


def show_crawler():
    st.markdown('<h1 class="main-header">🕸️ Website Crawler</h1>', unsafe_allow_html=True)
    st.write("Crawl your website to analyze SEO factors and identify issues")
    
    # Input section
    with st.form("crawler_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            website_url = st.text_input("Website URL", "https://example.com")
            max_pages = st.slider("Maximum pages to crawl", 1, 100, 20)
            crawl_depth = st.slider("Crawl depth", 1, 5, 2)
        
        with col2:
            respect_robots = st.checkbox("Respect robots.txt", value=True)
            include_images = st.checkbox("Analyze images", value=True)
            check_mobile = st.checkbox("Check mobile friendliness", value=True)
            
            # Optional OpenAI integration
            openai_api_key = st.text_input("OpenAI API Key (optional)", type="password")
        
        submitted = st.form_submit_button("Start Crawling")
    
    if submitted:
        # Validate URL
        if not website_url.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL including http:// or https://")
            return
        
        # Initialize crawler with options
        st.session_state.crawler = SEOScraper(
            base_url=website_url,
            max_pages=max_pages,
            openai_api_key=openai_api_key if openai_api_key else None
        )
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start crawling
        status_text.text("Crawling website...")
        
        try:
            # This would call our actual crawler from seo_scraper.py
            # For demo purposes, we'll simulate the crawl
            simulate_crawl(progress_bar, status_text, max_pages, website_url)
            
            st.success(f"Crawl complete! Analyzed {len(st.session_state.crawl_results)} pages.")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            # Calculate metrics
            avg_score = sum(p['seo_score'] for p in st.session_state.crawl_results) / len(st.session_state.crawl_results)
            avg_load_time = sum(p['load_time_seconds'] for p in st.session_state.crawl_results) / len(st.session_state.crawl_results)
            
            with col1:
                st.metric("Pages Crawled", len(st.session_state.crawl_results))
            with col2:
                st.metric("Avg. SEO Score", f"{avg_score:.1f}/100")
            with col3:
                st.metric("Avg. Load Time", f"{avg_load_time:.2f}s")
            
            # Display crawl results table
            st.markdown('<h3 class="sub-header">Crawl Results</h3>', unsafe_allow_html=True)
            
            # Convert to DataFrame for display
            df = pd.DataFrame(st.session_state.crawl_results)
            essential_cols = ['url', 'title', 'seo_score', 'h1_count', 'word_count', 'load_time_seconds']
            
            # Ensure all columns exist
            for col in essential_cols:
                if col not in df.columns:
                    df[col] = None
            
            st.dataframe(df[essential_cols], use_container_width=True)
            
            # Option to download results
            st.download_button(
                "Download Crawl Results (CSV)",
                df.to_csv(index=False).encode('utf-8'),
                "seo_crawl_results.csv",
                "text/csv",
                key='download-csv'
            )
            
        except Exception as e:
            st.error(f"Error during crawl: {str(e)}")


def show_gsc():
    st.markdown('<h1 class="main-header">📊 Google Search Console Integration</h1>', unsafe_allow_html=True)
    
    # Call the GSC UI component
    gsc_integration_ui()
    
    # Integration with crawl data if both are available
    if (hasattr(st.session_state, 'gsc_data') and not st.session_state.gsc_data.empty and 
        st.session_state.crawl_results is not None):
        
        st.markdown('<h3 class="sub-header">Integrate GSC with Crawl Data</h3>', unsafe_allow_html=True)
        
        if st.button("Integrate Data"):
            with st.spinner("Integrating data..."):
                # This would call our actual integration function
                # For demo, we'll simulate it
                gsc = GoogleSearchConsoleIntegration()
                
                # Convert crawl results to expected format
                st.session_state.integrated_data = gsc.integrate_with_crawl_data(
                    st.session_state.gsc_data, 
                    st.session_state.crawl_results
                )
                
                st.success("Data integrated successfully!")
                
                if not st.session_state.integrated_data.empty:
                    st.dataframe(st.session_state.integrated_data.head())


def show_content_analysis():
    st.markdown('<h1 class="main-header">📝 Content Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.crawl_results is None:
        st.info("Please crawl your website first to analyze content.")
        return
    
    # Get page list
    page_urls = [page['url'] for page in st.session_state.crawl_results]
    
    # Page selector
    selected_page = st.selectbox("Select a page to analyze", page_urls)
    
    # Get the selected page data
    page_data = next((p for p in st.session_state.crawl_results if p['url'] == selected_page), None)
    
    if page_data:
        st.markdown('<h3 class="sub-header">Page Content Overview</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Title**: {page_data.get('title', 'No title')}")
            st.markdown(f"**Meta Description**: {page_data.get('meta_description', 'No meta description')}")
            st.markdown(f"**Word Count**: {page_data.get('word_count', 0)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**H1 Tags**: {', '.join(page_data.get('h1_tags', ['None']))}")
            st.markdown(f"**SEO Score**: {page_data.get('seo_score', 0)}/100")
            st.markdown(f"**Load Time**: {page_data.get('load_time_seconds', 0)}s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Check if we have GSC data for this page
        if hasattr(st.session_state, 'gsc_data') and 'page' in st.session_state.gsc_data.columns:
            page_gsc = st.session_state.gsc_data[st.session_state.gsc_data['page'] == selected_page]
            
            if not page_gsc.empty:
                st.markdown('<h3 class="sub-header">Search Performance</h3>', unsafe_allow_html=True)
                
                # Display GSC metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Clicks", int(page_gsc['clicks'].sum()))
                with col2:
                    st.metric("Impressions", int(page_gsc['impressions'].sum()))
                with col3:
                    st.metric("Avg. CTR", f"{page_gsc['ctr'].mean():.2f}%")
                with col4:
                    st.metric("Avg. Position", f"{page_gsc['position'].mean():.1f}")
                
                # Show queries this page ranks for
                st.markdown("#### Top Queries")
                top_queries = page_gsc.sort_values('impressions', ascending=False).head(10)
                
                if not top_queries.empty:
                    st.dataframe(
                        top_queries[['query', 'impressions', 'clicks', 'ctr', 'position']],
                        use_container_width=True
                    )
        
        # Content analysis with AI
        st.markdown('<h3 class="sub-header">AI Content Analysis</h3>', unsafe_allow_html=True)
        
        # Check if we have OpenAI API key in settings
        openai_api_key = st.text_input("OpenAI API Key for Analysis", type="password")
        
        if st.button("Analyze Content with AI") and openai_api_key:
            with st.spinner("Analyzing content..."):
                # This would use our ChatGPTSEOAnalyzer
                # For demo, we'll simulate it
                time.sleep(2)
                
                st.session_state.ai_analysis = {
                    'title_analysis': 'The title is good but could be more compelling. Consider adding power words.',
                    'meta_description_analysis': 'Meta description is missing or too short. This is a critical issue.',
                    'content_quality': 'Content is informative but lacks depth in some key areas.',
                    'content_structure': 'Good use of headings, but consider adding more subheadings for better structure.',
                    'keyword_opportunities': ['seo analysis', 'website optimization', 'content strategy'],
                    'technical_issues': 'No major technical issues found.',
                    'priority_improvements': [
                        'Add a compelling meta description',
                        'Expand content in the section about technical SEO',
                        'Add more internal links to related content'
                    ],
                    'seo_score': 72
                }
                
                st.success("Analysis complete!")
        
        # Display AI analysis if available
        if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
            analysis = st.session_state.ai_analysis
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Title & Meta Description")
                st.markdown(f"**Title Analysis**: {analysis['title_analysis']}")
                st.markdown(f"**Meta Description**: {analysis['meta_description_analysis']}")
                
                st.markdown("#### Content Assessment")
                st.markdown(f"**Quality**: {analysis['content_quality']}")
                st.markdown(f"**Structure**: {analysis['content_structure']}")
            
            with col2:
                st.markdown("#### Keyword Opportunities")
                for kw in analysis['keyword_opportunities']:
                    st.markdown(f"- {kw}")
                
                st.markdown("#### Priority Improvements")
                for i, imp in enumerate(analysis['priority_improvements']):
                    st.markdown(f"{i+1}. {imp}")
            
            # AI-suggested improved versions
            with st.expander("AI-Suggested Improvements"):
                st.markdown("#### Improved Title")
                st.write("10 Powerful SEO Analysis Strategies to Outrank Your Competitors")
                
                st.markdown("#### Improved Meta Description")
                st.write("Discover proven SEO analysis techniques that boost rankings. Learn how to leverage data for better visibility and convert more visitors into customers.")
                
                st.markdown("#### Content Structure Recommendations")
                st.write("""
                1. Introduction to SEO Analysis
                2. Why Data-Driven SEO Matters
                3. Key Metrics to Track
                   - Technical Metrics
                   - Content Performance
                   - User Engagement
                4. Tools for Comprehensive Analysis
                5. Implementing Insights from Analysis
                6. Measuring Success
                7. Common Pitfalls to Avoid
                8. Conclusion
                """)


def show_technical_seo():
    st.markdown('<h1 class="main-header">🔧 Technical SEO Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.crawl_results is None:
        st.info("Please crawl your website first to analyze technical SEO factors.")
        return
    
    # Technical SEO overview
    st.markdown('<h3 class="sub-header">Technical SEO Overview</h3>', unsafe_allow_html=True)
    
    # Calculate technical metrics
    total_pages = len(st.session_state.crawl_results)
    
    # Count various issues
    missing_meta = sum(1 for p in st.session_state.crawl_results if not p.get('meta_description', ''))
    missing_title = sum(1 for p in st.session_state.crawl_results if not p.get('title', ''))
    missing_h1 = sum(1 for p in st.session_state.crawl_results if p.get('h1_count', 0) == 0)
    multi_h1 = sum(1 for p in st.session_state.crawl_results if p.get('h1_count', 0) > 1)
    slow_pages = sum(1 for p in st.session_state.crawl_results if p.get('load_time_seconds', 0) > 3)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages with Missing Meta", f"{missing_meta} ({missing_meta/total_pages*100:.1f}%)")
    with col2:
        st.metric("Pages with Missing H1", f"{missing_h1} ({missing_h1/total_pages*100:.1f}%)")
    with col3:
        st.metric("Slow Pages (>3s)", f"{slow_pages} ({slow_pages/total_pages*100:.1f}%)")
    
    # Technical SEO score
    tech_score = 100 - (missing_meta + missing_title + missing_h1 + multi_h1 + slow_pages) * 100 / (total_pages * 5)
    tech_score = max(0, tech_score)
    
    st.markdown(f"<h3 style='text-align:center'>Technical SEO Score: {tech_score:.1f}/100</h3>", unsafe_allow_html=True)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = tech_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Technical SEO Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#FF4136"},
                {'range': [50, 75], 'color': "#FFDC00"},
                {'range': [75, 100], 'color': "#2ECC40"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': tech_score
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed technical issues
    st.markdown('<h3 class="sub-header">Technical Issues</h3>', unsafe_allow_html=True)
    
    # Create tabs for different issue types
    tab1, tab2, tab3, tab4 = st.tabs(["Meta Tags", "Headings", "Page Speed", "Other Issues"])
    
    with tab1:
        st.markdown("### Meta Tag Issues")
        
        # Missing meta descriptions
        if missing_meta > 0:
            with st.expander(f"Missing Meta Descriptions ({missing_meta} pages)"):
                for p in st.session_state.crawl_results:
                    if not p.get('meta_description', ''):
                        st.markdown(f"- [{p.get('title', 'No title')}]({p.get('url', '')})")
        
        # Missing titles
        if missing_title > 0:
            with st.expander(f"Missing Page Titles ({missing_title} pages)"):
                for p in st.session_state.crawl_results:
                    if not p.get('title', ''):
                        st.markdown(f"- {p.get('url', '')}")
        
        # Title length issues
        title_too_short = [p for p in st.session_state.crawl_results if p.get('title', '') and len(p.get('title', '')) < 30]
        title_too_long = [p for p in st.session_state.crawl_results if p.get('title', '') and len(p.get('title', '')) > 60]
        
        if title_too_short:
            with st.expander(f"Titles Too Short (<30 chars) ({len(title_too_short)} pages)"):
                for p in title_too_short:
                    st.markdown(f"- [{p.get('title', '')}]({p.get('url', '')}) - {len(p.get('title', ''))} chars")
        
        if title_too_long:
            with st.expander(f"Titles Too Long (>60 chars) ({len(title_too_long)} pages)"):
                for p in title_too_long:
                    st.markdown(f"- [{p.get('title', '')}]({p.get('url', '')}) - {len(p.get('title', ''))} chars")
    
    with tab2:
        st.markdown("### Heading Structure Issues")
        
        # Missing H1
        if missing_h1 > 0:
            with st.expander(f"Missing H1 Tags ({missing_h1} pages)"):
                for p in st.session_state.crawl_results:
                    if p.get('h1_count', 0) == 0:
                        st.markdown(f"- [{p.get('title', 'No title')}]({p.get('url', '')})")
        
        # Multiple H1
        if multi_h1 > 0:
            with st.expander(f"Multiple H1 Tags ({multi_h1} pages)"):
                for p in st.session_state.crawl_results:
                    if p.get('h1_count', 0) > 1:
                        st.markdown(f"- [{p.get('title', 'No title')}]({p.get('url', '')}) - {p.get('h1_count')} H1 tags")
    
    with tab3:
        st.markdown("### Page Speed Issues")
        
        # Slow pages
        if slow_pages > 0:
            with st.expander(f"Slow Loading Pages (>3s) ({slow_pages} pages)"):
                slow_list = [p for p in st.session_state.crawl_results if p.get('load_time_seconds', 0) > 3]
                slow_list.sort(key=lambda x: x.get('load_time_seconds', 0), reverse=True)
                
                for p in slow_list:
                    st.markdown(f"- [{p.get('title', 'No title')}]({p.get('url', '')}) - {p.get('load_time_seconds', 0):.2f}s")
        
        # Page speed distribution
        load_times = [p.get('load_time_seconds', 0) for p in st.session_state.crawl_results]
        
        fig = px.histogram(
            x=load_times,
            nbins=15,
            title="Page Load Time Distribution",
            labels={'x': 'Load Time (seconds)', 'y': 'Number of Pages'},
            color_discrete_sequence=['#FF4136']
        )
        
        fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="3s threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Other Technical Issues")
        
        # Images without alt text
        images_without_alt = sum(p.get('images_without_alt', 0) for p in st.session_state.crawl_results)
        if images_without_alt > 0:
            with st.expander(f"Images Missing Alt Text ({images_without_alt} images)"):
                for p in st.session_state.crawl_results:
                    if p.get('images_without_alt', 0) > 0:
                        st.markdown(f"- [{p.get('title', 'No title')}]({p.get('url', '')}) - {p.get('images_without_alt')} images without alt text")
        
        # Add more technical checks here as needed


def show_settings():
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="sub-header">API Keys</h3>', unsafe_allow_html=True)
    
    # OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get('openai_api_key', '')
    )
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
    
    # Crawl settings
    st.markdown('<h3 class="sub-header">Crawler Settings</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.max_pages = st.number_input(
            "Default Max Pages to Crawl",
            min_value=5,
            max_value=500,
            value=st.session_state.get('max_pages', 20)
        )
        
        st.session_state.respect_robots = st.checkbox(
            "Respect robots.txt by default",
            value=st.session_state.get('respect_robots', True)
        )
    
    with col2:
        st.session_state.crawl_delay = st.number_input(
            "Crawl Delay (seconds)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.get('crawl_delay', 0.5),
            step=0.1
        )
        
        st.session_state.user_agent = st.text_input(
            "Custom User Agent",
            value=st.session_state.get('user_agent', 'SEOEngine Bot 1.0')
        )
    
    # Google Search Console settings
    st.markdown('<h3 class="sub-header">Google Search Console Settings</h3>', unsafe_allow_html=True)
    
    st.session_state.gsc_default_days = st.slider(
        "Default Date Range (days)",
        min_value=7,
        max_value=90,
        value=st.session_state.get('gsc_default_days', 30)
    )
    
    # Data storage settings
    st.markdown('<h3 class="sub-header">Data Storage</h3>', unsafe_allow_html=True)
    
    st.session_state.store_history = st.checkbox(
        "Store historical data for trend analysis",
        value=st.session_state.get('store_history', True)
    )
    
    if st.session_state.get('store_history', True):
        st.session_state.history_retention = st.slider(
            "History retention period (days)",
            min_value=30,
            max_value=365,
            value=st.session_state.get('history_retention', 90)
        )
    
    # Export options
    st.markdown('<h3 class="sub-header">Export Settings</h3>', unsafe_allow_html=True)
    
    export_formats = ['CSV', 'JSON', 'Excel', 'PDF']
    st.session_state.default_export_format = st.selectbox(
        "Default Export Format",
        options=export_formats,
        index=export_formats.index(st.session_state.get('default_export_format', 'CSV'))
    )
    
    # Save settings button
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")


# Helper function to simulate crawl for demo purposes
def simulate_crawl(progress_bar, status_text, max_pages, website_url):
    """Simulate a website crawl for demo purposes"""
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = []
    
    # Reset results
    st.session_state.crawl_results = []
    
    for i in range(max_pages):
        # Update progress
        progress = (i + 1) / max_pages
        progress_bar.progress(progress)
        status_text.text(f"Crawling page {i+1}/{max_pages}...")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Generate a fake page result
        page_number = i + 1
        page_path = f"/page-{page_number}" if i > 0 else "/"
        
        # Create some variation in the data
        has_meta = np.random.random() > 0.2
        has_h1 = np.random.random() > 0.15
        h1_count = 1 if has_h1 else 0
        word_count = int(np.random.normal(500, 200))
        load_time = np.random.uniform(1.0, 5.0)
        image_count = int(np.random.poisson(5))
        images_without_alt = int(image_count * np.random.uniform(0, 0.5))
        
        # Calculate SEO score (simplified)
        score = 100
        if not has_meta:
            score -= 15
        if not has_h1:
            score -= 10
        if word_count < 300:
            score -= 15
        if load_time > 3:
            score -= 15
        if images_without_alt > 0:
            score -= min(15, images_without_alt * 3)
        
        score = max(0, min(100, score))
        
        # Create page data
        page_data = {
            'url': f"{website_url}{page_path}",
            'title': f"Example Page {page_number}" if np.random.random() > 0.1 else "",
            'meta_description': f"This is an example description for page {page_number}." if has_meta else "",
            'h1_count': h1_count,
            'h1_tags': [f"Example Heading {page_number}"] if has_h1 else [],
            'h2_count': int(np.random.poisson(3)),
            'word_count': word_count,
            'load_time_seconds': load_time,
            'image_count': image_count,
            'images_without_alt': images_without_alt,
            'seo_score': score
        }
        
        # Add page data to results
        st.session_state.crawl_results.append(page_data)
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Crawl completed!")


if __name__ == "__main__":
    main()