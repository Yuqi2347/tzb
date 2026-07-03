document.addEventListener('DOMContentLoaded', () => {
    // 健康检查轮询
    const checkHealth = async () => {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            const loadingOverlay = document.getElementById('loading-overlay');
            if (data.initialized) {
                loadingOverlay.classList.add('hidden');
            } else if (data.initialization_error) {
                loadingOverlay.classList.add('hidden');
                document.body.innerHTML = `<div class="text-red-500 text-center mt-10">初始化失败: ${data.initialization_error}</div>`;
            }
        } catch (error) {
            console.error('健康检查失败:', error);
        }
    };
    checkHealth();
    setInterval(checkHealth, 3000);

    // ==================== 1. 新增: 获取新按钮 & 存储初始状态 ====================
    const newSessionBtn = document.getElementById('new-session-btn');
    const locateToggle = document.getElementById('locate-mode-toggle');
    let locateMode = false;
    function updateLocateToggleUI() {
        if (!locateToggle) return;
        const label = locateToggle.querySelector('span');
        if (locateMode) {
            label.textContent = '定位模式：开';
            locateToggle.classList.add('bg-black', 'text-white');
            locateToggle.classList.remove('text-text-secondary');
        } else {
            label.textContent = '定位模式：关';
            locateToggle.classList.remove('bg-black', 'text-white');
            locateToggle.classList.add('text-text-secondary');
        }
    }
    updateLocateToggleUI();
    locateToggle?.addEventListener('click', () => { locateMode = !locateMode; updateLocateToggleUI(); });

    const chatContainer = document.getElementById('chat-container');
    const initialChatHTML = chatContainer.innerHTML; // 保存初始的欢迎界面HTML

    let chatHistory = [];
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const uploadBtn = document.getElementById('upload-btn');
    const imageUpload = document.getElementById('image-upload');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImage = document.getElementById('preview-image');
    const previewFilename = document.getElementById('preview-filename');
    const removeImage = document.getElementById('remove-image');
    let selectedImage = null;

    // ==================== 2. 新增: 定义并绑定新会话函数 ====================
    function startNewSession() {
        // (A) 重置对话历史记录
        chatHistory = [];
        // (B) 恢复聊天窗口的初始HTML（即欢迎界面）
        chatContainer.innerHTML = initialChatHTML;
        // (C) 清空输入框
        messageInput.value = '';
        messageInput.style.height = 'auto';
        // (D) 如果有图片预览，也清空
        removeImage.click(); // 触发removeImage按钮的点击事件，重用其逻辑
    }

    newSessionBtn.addEventListener('click', startNewSession);
    // ======================================================================

    // 调整文本框高度
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        const newHeight = Math.min(messageInput.scrollHeight, 200);
        messageInput.style.height = `${newHeight}px`;
    });

    function handleImageFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            console.warn('请上传图片文件!');
            return;
        }
        selectedImage = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewFilename.textContent = file.name;
            uploadPreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    uploadBtn.addEventListener('click', () => imageUpload.click());
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleImageFile(e.target.files[0]);
    });

    const dropZone = document.body;
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); e.stopPropagation(); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const files = e.dataTransfer.files;
        if (files.length > 0) handleImageFile(files[0]);
    });

    removeImage.addEventListener('click', () => {
        selectedImage = null;
        imageUpload.value = '';
        uploadPreview.classList.add('hidden');
    });

    // addMessageToChat 函数无变化
    function addMessageToChat(content, isUser = false) {
        document.getElementById('welcome-screen')?.remove();

        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'w-full flex mb-6';

        if (isUser) {
            messageWrapper.classList.add('justify-end');
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'flex items-start gap-4 max-w-2xl';

        if (isUser) {
            messageContent.classList.add('flex-row-reverse');
        }

        const avatar = document.createElement('div');
        avatar.className = 'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-white font-semibold';
        if (isUser) {
            avatar.classList.add('bg-user-avatar-bg');
            avatar.textContent = 'U';
        } else {
            avatar.classList.add('bg-ai-avatar-bg');
            const icon = document.createElement('i');
            icon.className = 'fa fa-robot';
            avatar.appendChild(icon);
        }

        const contentContainer = document.createElement('div');
        contentContainer.className = 'flex flex-col pt-1';

        if (isUser && selectedImage) {
            const imgElement = document.createElement('img');
            imgElement.src = previewImage.src;
            imgElement.className = 'w-48 object-cover rounded-lg mb-2';
            contentContainer.appendChild(imgElement);
        }

        const textParagraph = document.createElement('p');
        textParagraph.className = 'text-text-primary leading-relaxed';
        textParagraph.textContent = content;
        contentContainer.appendChild(textParagraph);

        messageContent.appendChild(avatar);
        messageContent.appendChild(contentContainer);
        messageWrapper.appendChild(messageContent);
        chatContainer.appendChild(messageWrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        return isUser ? null : contentContainer;
    }

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }
    });

    // submit 事件监听函数无变化
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text && !selectedImage) return;

        chatHistory.push({ role: 'user', content: text });
        addMessageToChat(text, true);

        const currentSelectedImage = selectedImage;
        selectedImage = null;

        messageInput.value = '';
        messageInput.style.height = 'auto';
        uploadPreview.classList.add('hidden');
        imageUpload.value = '';

        const aiMessageContainer = addMessageToChat('', false);
        const typingIndicator = document.createElement('div');
        typingIndicator.textContent = '...';
        typingIndicator.className = 'text-2xl text-text-secondary animate-pulse';
        aiMessageContainer.appendChild(typingIndicator);

        try {
            const formData = new FormData();
            formData.append('text', text);
            formData.append('history_json', JSON.stringify(chatHistory.slice(0, -1)));
            formData.append('locate_mode', locateMode ? '1' : '0');
            if (currentSelectedImage) {
                formData.append('image', currentSelectedImage, currentSelectedImage.name);
            }

            const response = await fetch('/api/chat', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('请求失败');

            if (locateMode) {
                let contentType = (response.headers.get('Content-Type') || response.headers.get('content-type') || '').toLowerCase();
                let isImage = contentType.startsWith('image/') || contentType.includes('application/octet-stream');
                let bufForImage = null;
                // 签名嗅探（PNG/JPEG），避免把图片二进制当作文本显示
                if (!isImage) {
                    const clone = response.clone();
                    const buf = await clone.arrayBuffer();
                    const head = new Uint8Array(buf).subarray(0, 8);
                    const isPNG = head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4E && head[3] === 0x47 &&
                                  head[4] === 0x0D && head[5] === 0x0A && head[6] === 0x1A && head[7] === 0x0A;
                    const isJPEG = head[0] === 0xFF && head[1] === 0xD8;
                    if (isPNG || isJPEG) {
                        isImage = true;
                        bufForImage = buf;
                    }
                }
                // 如果是图片响应，直接渲染图片
                if (isImage) {
                    typingIndicator.remove();
                    let blob;
                    if (bufForImage) {
                        const t = contentType && contentType.startsWith('image/') ? contentType : 'image/png';
                        blob = new Blob([bufForImage], { type: t });
                    } else {
                        blob = await response.blob();
                    }
                    const img = document.createElement('img');
                    img.src = URL.createObjectURL(blob);
                    img.className = 'w-64 object-cover rounded-lg mb-2 border border-border-color';
                    aiMessageContainer.appendChild(img);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    chatHistory.push({ role: 'assistant', content: '[图像定位结果]' });
                    return;
                }
                // 定位模式下：非图片就读纯文本（错误提示）
                const errText = await response.text();
                typingIndicator.remove();
                const errorP = document.createElement('p');
                errorP.className = 'text-red-500';
                errorP.textContent = errText || '定位失败';
                aiMessageContainer.appendChild(errorP);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                chatHistory.push({ role: 'assistant', content: errText || '定位失败' });
                return;
            }

            // 正常模式：按文本流式处理
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            typingIndicator.remove();
            const aiResponseText = document.createElement('p');
            aiResponseText.className = 'text-text-primary leading-relaxed';
            aiMessageContainer.appendChild(aiResponseText);

            let fullResponse = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                fullResponse += chunk;
                aiResponseText.textContent = fullResponse;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            chatHistory.push({ role: 'assistant', content: fullResponse });
        } catch (error) {
            typingIndicator.remove();
            const errorP = document.createElement('p');
            errorP.className = 'text-red-500';
            errorP.textContent = '请求失败，请重试';
            aiMessageContainer.appendChild(errorP);
            chatHistory.pop();
        }
    });
});