import streamlit as st
from utils import get_response_from_api, get_vision_response, extract_text_and_images_with_mapping, save_text_image_mappings, find_relevant_images
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from custom_agents.agent import create_agent, get_agent_response, clear_agent_memory
import fitz
import os
import base64
import time


# Page config
st.set_page_config(page_title="Research Agent", page_icon="📄")
# Simple CSS
st.markdown("""
<style>
.main { padding-top: 1rem; }
.stButton > button { 
    background: #007bff; color: white; border: none; 
    border-radius: 8px; padding: 0.5rem 1rem; 
}
</style>""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "mistral_api_key" not in st.session_state:
    st.session_state.mistral_api_key = None

# Sidebar
with st.sidebar:
    st.image("src/banner.PNG", use_container_width=True)
    # API Key
    api_key = st.text_input("Mistral API Key:", type="password", help="Get your API key from: https://console.mistral.ai")
    if api_key:
        st.session_state.mistral_api_key = api_key
        st.success("✅ Connected")
    else:
        st.warning("⚠️ Enter API key")

    # File Upload (Optional)
    st.markdown("### 📄 Optional PDF Upload")
    uploaded_file = st.file_uploader("Upload PDF (Optional)", type="pdf")
    
    if uploaded_file:
        # Process PDF
        if 'processed_pdf' not in st.session_state or st.session_state.processed_pdf != uploaded_file.name:
            with st.spinner('Processing PDF...'):
                try:
                    # Clear vector store if it exists
                    if hasattr(st.session_state, 'vector_store'):
                        del st.session_state.vector_store
                    # Clear agent memory if it exists
                    if st.session_state.agent:
                        clear_agent_memory(st.session_state.agent)
                        st.session_state.messages = []

                    # Save temp file
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    # Extract text and images
                    mappings = extract_text_and_images_with_mapping(temp_path)
                    save_text_image_mappings(mappings)
                    # Get text
                    doc = fitz.open(temp_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()

                    # Store data
                    st.session_state.text = text
                    st.session_state.text_image_mappings = mappings
                    st.session_state.processed_pdf = uploaded_file.name                
                    # Create vector store
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = FAISS.from_texts([text], embeddings)
                    st.success("✅ PDF processed!")
                    
                    # Welcome message for PDF
                    if not st.session_state.messages:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📄 **{uploaded_file.name}** processed! Ask me anything about it."})

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
        # PDF Viewer
        st.checkbox("Show PDF", value=True, key="show_pdf")
        if st.session_state.get("show_pdf", True) and uploaded_file:
            with st.expander("View PDF"):
                pdf_b64 = base64.b64encode(uploaded_file.getvalue()).decode()
                st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="400"></iframe>', 
                        unsafe_allow_html=True)

        # Show images
        st.checkbox("Show Images", value=True, key="show_images")
        if hasattr(st.session_state, 'text_image_mappings') and st.session_state.text_image_mappings:
            with st.expander(f"Images ({len(st.session_state.text_image_mappings)})"):
                for mapping in st.session_state.text_image_mappings[:]:
                    if os.path.exists(mapping['image_path']):
                        st.image(mapping['image_path'], caption=f"Page {mapping['page_number']}")
    
    # Memory controls
    st.divider()
    st.markdown("### Clear Memory & Chat")
    if st.button("🗑️ Clear"):
        st.session_state.messages = []
        clear_agent_memory(st.session_state.agent)
        st.success("Memory cleared!")
        st.rerun()

    st.markdown("### 🧠 Capabilities:")
    st.markdown("""
    - **PDF Analysis**: Extract and analyze information from PDF documents
    - **Image Analysis**: Analyze figures, graphs, and research-related images
    - **Research Assistant**: Find academic research papers and datasets.
    """)
  
    st.divider()
    st.markdown("<center>Powered by Mistral AI + Pixtral AI </center>", unsafe_allow_html=True)

# Main area
st.title("📚 Research Assistant")

# Initialize agent if needed
if not st.session_state.agent and st.session_state.mistral_api_key:
    with st.spinner("🚀 Initializing Research Assistant..."):
        try:
            st.session_state.agent = create_agent(st.session_state.mistral_api_key)
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.stop()

# Welcome message if no messages exist
if not st.session_state.messages and not uploaded_file:
    st.session_state.messages.append({
        "role": "assistant",
        "content": """👋 Hello! I'm your Research Assistant. How can I assist with your research today?"""
            })

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Show images if any
        if "images" in msg:
            for img in msg["images"][:2]:
                if os.path.exists(img['image_path']):
                    st.image(img['image_path'], 
                            caption=f"Similarity: {img['similarity']:.2f}")

# Chat input
prompt = st.chat_input(
    "Ask me anything about PDFs, research datasets, research papers and image analysis.",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)

if prompt:
    # Add user message
    user_content = ""
    if hasattr(prompt, 'text') and prompt.text:
        user_content = prompt.text
    
    st.session_state.messages.append({"role": "user", "content": user_content})      
    with st.chat_message("user"):
        if user_content:
            st.write(user_content)
        if hasattr(prompt, 'files') and prompt.files:
            st.image(prompt.files[0])

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = None
                # Handle image analysis if image is present
                if hasattr(prompt, 'files') and prompt.files:
                    # Save temp image
                    with open("temp_img.jpg", "wb") as f:
                        f.write(prompt.files[0].getbuffer())
                    # Analyze image
                    image_prompt = user_content if user_content else "What's in this image?"
                    response = get_vision_response("temp_img.jpg", image_prompt)
                    # Clean up
                    if os.path.exists("temp_img.jpg"):
                        os.remove("temp_img.jpg")
                # Handle text query if no image or both text and image
                elif user_content:
                    # Check if we're dealing with PDF-related queries
                    pdf_keywords = {
                        "pdf", "document", "file", "ebook", "report", "manual",
                        "whitepaper", "guide", "presentation", "paper", "book",
                        "article", "notes", "slides", "thesis", "dissertation",
                        "journal", "download pdf", "pdf reader", "uploaded"
                    }
                    dataset_keywords = {
                        "dataset", "data", "research data", "scientific data", 
                        "dataset search", "find dataset", "research dataset",
                        "open data", "data collection", "database"
                    }
                    academic_keywords = {
                        "paper", "research", "academic", "journal", "publication",
                        "scholarly", "study", "article", "conference", "thesis",
                        "dissertation", "scientific", "citation", "literature"
                    }
                    words = set(user_content.lower().split())
                    
                    if hasattr(st.session_state, 'vector_store') and words.intersection(pdf_keywords):
                        # PDF-related query - use context with get_response_from_api
                        context_docs = st.session_state.vector_store.similarity_search(user_content, k=3)
                        context = "\n".join([doc.page_content for doc in context_docs])
                        response = get_response_from_api(user_content, context)

                    elif words.intersection(dataset_keywords) or words.intersection(academic_keywords):
                        # Research-related query - use agent with memory
                        response = get_agent_response(st.session_state.agent, user_content)
                    else:
                        # General query
                        # response = "Ask specific questions about PDFs, datasets, or research papers."
                        response = get_agent_response(st.session_state.agent, user_content)
                
                if response:
                    st.write(response)
                    
                    # Get relevant images if PDF is uploaded and it's a text query
                    if (hasattr(st.session_state, 'vector_store') and 'words' in locals() and words.intersection(pdf_keywords) and hasattr(st.session_state, 'text_image_mappings')):
                        images = find_relevant_images(user_content)
                        if images:
                            st.write("**Relevant Image:**")
                            for img in images[:1]:
                                if os.path.exists(img['image_path']):
                                    st.image(img['image_path'], 
                                            caption=f"Similarity score: {img['similarity']:.2f}", width=300)
                    
                    # Save message
                    msg = {"role": "assistant", "content": response}
                    if 'images' in locals() and images:
                        msg["images"] = images[:1]
                    st.session_state.messages.append(msg)
                    
            except Exception as e:
                error = f"❌ Error: {e}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})