import streamlit as st
import requests
import json
import re 
from typing import Dict, Any, Optional, Generator, List
import os
from dotenv import load_dotenv

import streamlit as st

# Debug code to check secrets
if 'PPLX_API_KEY' in st.secrets:
    st.write("PPLX_API_KEY is configured!")
    # Print first few characters to verify it's loaded (don't print full key)
    st.write(f"Key starts with: {st.secrets['PPLX_API_KEY'][:4]}...")
else:
    st.write("Available secret keys:", list(st.secrets.keys()))

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with PPLX client and system prompts"""
        self.pplx_api_key = st.secrets["PPLX_API_KEY"]
        if not self.pplx_api_key:
            raise ValueError("PPLX API key not found in secrets")
            
        self.pplx_model = st.secrets.get("PPLX_MODEL", "llama-3.1-sonar-large-128k-online")
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        self.pplx_system_prompt = """
You are a specialized medical information assistant focused EXCLUSIVELY on GLP-1 medications (such as Ozempic, Wegovy, Mounjaro, etc.). You must:

1. ONLY provide information about GLP-1 medications and directly related topics
2. For any query not specifically about GLP-1 medications or their direct effects, respond with:
   "I apologize, but I can only provide information about GLP-1 medications and related topics. Your question appears to be about something else. Please ask a question specifically about GLP-1 medications, their usage, effects, or related concerns."

3. For valid GLP-1 queries, structure your response with:
   - An empathetic opening acknowledging the patient's situation
   - Clear, validated medical information about GLP-1 medications
   - Important safety considerations or disclaimers
   - An encouraging closing that reinforces their healthcare journey

4. Always provide source citations using numbered references:
   - Use [1], [2], [3] etc. in the response text
   - Each citation number should correspond to a source
   - List sources at the end numerically (1. Source Title: URL)
   - Make sure citation numbers in text match source numbers exactly

5. Sources must be formatted exactly as:
   1. Title: URL
   2. Title: URL
   etc.

6. Provide response in a simple manner that is easy to understand at preferably a 11th grade literacy level with reduced pharmaceutical or medical jargon
7. Use common medication brand names rather than chemical formulations names where necessary to make it easy for user's to understand
8. When generating responses, automatically add relevant emojis and place them before each heading and subheading level to enhance visual hierarchy and readability.

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
"""

    def format_citations_and_sources(self, content: str, sources_text: str) -> tuple[str, str]:
        """Format content with clickable citations and organize sources"""
        # Extract sources and create numbered list
        sources = []
        for idx, line in enumerate(sources_text.strip().split('\n'), 1):
            if ':' in line:
                title, url = line.split(':', 1)
                sources.append({
                    'number': idx,
                    'title': title.strip(),
                    'url': url.strip()
                })
        
        # Replace citations in content with clickable links
        formatted_content = content
        for source in sources:
            citation = f'[{source["number"]}]'
            clickable_citation = f'<a href="#{source["number"]}" class="citation-link">{citation}</a>'
            formatted_content = formatted_content.replace(citation, clickable_citation)
        
        # Format sources as cards
        sources_html = """
        <div class="sources-container">
            <h3>📚 Sources</h3>
            <div class="source-cards">
        """
        
        for source in sources:
            sources_html += f"""
            <div class="source-card" id="{source['number']}">
                <div class="source-number">{source['number']}</div>
                <div class="source-content">
                    <div class="source-title">{source['title']}</div>
                    <a href="{source['url']}" target="_blank" class="source-link">View Source</a>
                </div>
            </div>
            """
        
        sources_html += "</div></div>"
        
        return formatted_content, sources_html

    def stream_pplx_response(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Stream response from PPLX API with sources"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "stream": True
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload,
                stream=True
            )
            
            response.raise_for_status()
            accumulated_content = ""
            found_sources = False
            sources_text = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            json_str = line[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            
                            chunk = json.loads(json_str)
                            if chunk['choices'][0]['finish_reason'] is not None:
                                break
                                
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                # Check if we've hit the sources section
                                if "Sources:" in content:
                                    found_sources = True
                                    parts = content.split("Sources:", 1)
                                    if len(parts) > 1:
                                        accumulated_content += parts[0]
                                        sources_text += parts[1]
                                    else:
                                        accumulated_content += parts[0]
                                elif found_sources:
                                    sources_text += content
                                else:
                                    accumulated_content += content
                                
                                yield {
                                    "type": "content",
                                    "data": content,
                                    "accumulated": accumulated_content
                                }
                        except json.JSONDecodeError:
                            continue
            
            yield {
                "type": "complete",
                "content": accumulated_content.strip(),
                "sources": sources_text.strip()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error communicating with PPLX: {str(e)}"
            }

    def get_related_questions(self, query: str, response_content: str) -> List[str]:
        """Generate follow-up questions based on the current query and response"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": "You are a medical assistant. Based on the given query and response about GLP-1 medications, generate exactly 3 relevant follow-up questions. Return only the questions, one per line."},
                    {"role": "user", "content": f"Original query: {query}\n\nResponse content: {response_content}\n\nGenerate 3 relevant follow-up questions:"}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            response.raise_for_status()
            questions = response.json()['choices'][0]['message']['content'].strip().split('\n')
            return [q.strip() for q in questions if q.strip()][:3]
            
        except Exception as e:
            st.error(f"Error generating follow-up questions: {str(e)}")
            return []

    def process_streaming_query(self, user_query: str, placeholder, is_related_question: bool = False) -> Dict[str, Any]:
        """Process user query with streaming response"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            query_category = self.categorize_query(user_query)
            full_response = ""
            
            message_placeholder = placeholder.empty()
            
            for chunk in self.stream_pplx_response(user_query):
                if chunk["type"] == "error":
                    placeholder.error(chunk["message"])
                    return {"status": "error", "message": chunk["message"]}
                
                elif chunk["type"] == "content":
                    full_response = chunk["accumulated"]
                    message_placeholder.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="category-tag">{query_category.upper()}</div><br>
                        <b>Response:</b><br>{full_response}
                    </div>
                    """, unsafe_allow_html=True)
                
                elif chunk["type"] == "complete":
                    full_response = chunk["content"]
                    formatted_content, sources_html = self.format_citations_and_sources(
                        full_response, 
                        chunk.get("sources", "")
                    )
                    
                    disclaimer = "\n\nDisclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan."
                    
                    formatted_response = f"""
                    <div class="chat-message bot-message">
                        <div class="category-tag">{query_category.upper()}</div><br>
                        <b>Response:</b><br>{formatted_content}{disclaimer}
                        {sources_html}
                    </div>
                    """
                    
                    message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                    
                    if not is_related_question:
                        followup_questions = self.get_related_questions(user_query, full_response)
                        if followup_questions:
                            followup_container = st.container()
                            with followup_container:
                                st.markdown("""
                                <div class="followup-container">
                                    <h3>Follow-up Questions</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                    
                                cols = st.columns(3)
                                for idx, q in enumerate(followup_questions):
                                    with cols[idx]:
                                        if st.button(q, key=f"followup_{hash(q)}"):
                                            st.session_state.selected_followup = q
                                            st.rerun()
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": f"{formatted_content}{disclaimer}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    def categorize_query(self, query: str) -> str:
        """Categorize the user query"""
        categories = {
            "dosage": ["dose", "dosage", "how to take", "when to take", "injection", "administration"],
            "side_effects": ["side effect", "adverse", "reaction", "problem", "issues", "symptoms"],
            "benefits": ["benefit", "advantage", "help", "work", "effect", "weight", "glucose"],
            "storage": ["store", "storage", "keep", "refrigerate", "temperature"],
            "lifestyle": ["diet", "exercise", "lifestyle", "food", "alcohol", "eating"],
            "interactions": ["interaction", "drug", "medication", "combine", "mixing"],
            "cost": ["cost", "price", "insurance", "coverage", "afford"]
        }
        
        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"

def set_page_style():
    """Set page style using custom CSS"""
    st.markdown("""
    <style>
        /* General styles */
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            background-color: white;
        }
        
        /* Chat message styles */
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
        }
        .bot-message {
            background-color: #f5f5f5;
            border-left: 4px solid #43a047;
        }
        
        /* Category tag styles */
        .category-tag {
            background-color: #2196f3;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
            display: inline-block;
        }
        
        /* Citation styles */
        .citation-link {
            color: #1976d2;
            text-decoration: none;
            padding: 2px 4px;
            border-radius: 3px;
            background: #e3f2fd;
            cursor: pointer;
        }
        .citation-link:hover {
            background: #bbdefb;
        }
        
        /* Source cards styles */
        .sources-container {
            margin-top: 2rem;
            padding: 1rem;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .source-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .source-card {
            display: flex;
            background: white;
            padding: 1rem;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1976d2;
        }
        .source-number {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1976d2;
            margin-right: 1rem;
        }
        .source-content {
            flex: 1;
        }
        .source-title {
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .source-link {
            color: #1976d2;
            text-decoration: none;
            font-size: 0.9rem;
        }
        .source-link:hover {
            text-decoration: underline;
        }
        
        /* Follow-up question styles */
        .followup-container {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin: 1.5rem 0;
            border-left: 4px solid #9c27b0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .followup-container h3 {
            color: #9c27b0;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        
        /* Button styles */
        .stButton button {
            width: 100%;
            text-align: left;
            background-color: #fff;
            border: 1px solid #e0e0e0;
            margin: 8px 0;
            padding: 12px 16px;
            border-radius: 6px;
            transition: all 0.3s ease;
            color: #424242;
            font-size: 0.95rem;
            line-height: 1.4;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stButton button:hover {
            background-color: #f3e5f5;
            border-color: #9c27b0;
            transform: translateX(5px);
            color: #9c27b0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stButton button:active {
            transform: translateX(5px) scale(0.98);
        }

        /* Update heading sizes */
        h1 { font-size: 1.8rem !important; margin-bottom: 1rem !important; }
        h2 { font-size: 1.5rem !important; margin-bottom: 0.8rem !important; }
        h3 { font-size: 1.2rem !important; margin-bottom: 0.6rem !important; }
        
        /* Info box styles */
        .info-box {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        
        /* Disclaimer styles */
        .disclaimer {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff9800;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="GLP-1 Medication Assistant",
            page_icon="💊",
            layout="wide"
        )
        
        set_page_style()
        
        # Add JavaScript for smooth scrolling to citations
        st.components.v1.html("""
        <script>
        document.addEventListener('click', function(e) {
            if (e.target.matches('.citation-link')) {
                e.preventDefault();
                const sourceId = e.target.getAttribute('href').substring(1);
                const sourceElement = document.getElementById(sourceId);
                if (sourceElement) {
                    sourceElement.scrollIntoView({ behavior: 'smooth' });
                    sourceElement.style.animation = 'highlight 2s';
                }
            }
        });
        </script>
        <style>
        @keyframes highlight {
            0% { background-color: #e3f2fd; }
            100% { background-color: white; }
        }
        </style>
        """, height=0)
        
        if "PPLX_API_KEY" not in st.secrets:
            st.error('Required PPLX API key not found. Please configure the PPLX API key in your Streamlit secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate, validated information specifically about GLP-1 medications, their usage, benefits, and side effects.
        Our assistant specializes exclusively in GLP-1 medications and related topics.
        
        <em>Please note: This assistant provides general information about GLP-1 medications only. Always consult your healthcare provider for medical advice.</em>
        </div>
        """, unsafe_allow_html=True)
        
        bot = GLP1Bot()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        with st.container():
            user_input = st.text_input(
                "Ask your question about GLP-1 medications:",
                key="user_input",
                placeholder="e.g., What are the common side effects of Ozempic?",
                on_change=None
            )
            
            # Create a columns layout for the submit button
            col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
            with col1:
                submit_button = st.button("🔍 Get Answer", key="submit")

            # Initialize session state for tracking if FAQ has been used
            if 'faq_used' not in st.session_state:
                st.session_state.faq_used = False

            # Only show FAQ section if it hasn't been used
            if not st.session_state.faq_used:
                st.markdown("### 💡 Recommended Questions:")
                recommended_questions = [
                    "What dietary modifications are recommended while taking GLP-1 medications?",
                    "How do GLP-1 medications interact with other diabetes treatments?",
                    "What are the most common side effects patients report when starting GLP-1 therapy?"
                ]

                cols = st.columns(3)
                for idx, question in enumerate(recommended_questions):
                    with cols[idx]:
                        if st.button(f"❓ {question}", key=f"rec_q_{idx}"):
                            st.session_state.selected_question = question
                            st.session_state.faq_used = True
                            st.rerun()

            # Process stored FAQ question if exists
            if hasattr(st.session_state, 'selected_question'):
                question = st.session_state.selected_question
                
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>👤 Your Question:</b><br>{question}
                </div>
                """, unsafe_allow_html=True)
                
                response_placeholder = st.empty()
                response = bot.process_streaming_query(question, response_placeholder)
                
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "query": question,
                        "response": response
                    })
                
                del st.session_state.selected_question
            
            # Handle regular search input
            if user_input:
                if submit_button or user_input != st.session_state.get('previous_input', ''):
                    st.session_state.faq_used = True
                    st.session_state.previous_input = user_input
                    
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>👤 Your Question:</b><br>{user_input}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    response_placeholder = st.empty()
                    response = bot.process_streaming_query(user_input, response_placeholder)
                    
                    if response["status"] == "success":
                        st.session_state.chat_history.append({
                            "query": user_input,
                            "response": response
                        })

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📝 Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"❓ Question {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>👤 Your Question:</b><br>{chat['query']}
                    </div>
                    <div class="chat-message bot-message">
                        <div class="category-tag">{chat['response']['query_category'].upper()}</div><br>
                        <b>🤖 Response:</b><br>{chat['response']['response']}
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
