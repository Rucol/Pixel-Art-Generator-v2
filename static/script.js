/**
 * Pixel Art Generator - Logic Controller
 */

let selectedTags = new Set();
let allTags = [];

/**
 * Category dictionary - maps JSON tags to UI dropdowns.
 * Used for grouping and exclusion logic.
 */
const CATEGORIES = {
    "GENDER": ["male", "female"],
    "HEAD": ["head_", "hair_"],
    "BODY": ["body_", "chest_", "torso_"],
    "ARMS": ["arms_", "hands_"],
    "LEGS": ["legs_", "feet_"],
    "WEAPON": ["weapon_", "sword_", "staff_", "bow_"],
    "SHIELD": ["shield_"],
    "OTHER": [] // Catch-all for remaining tags
};

/**
 * Default tags used if the user leaves a core category empty.
 * Ensures the GAN model always receives a coherent character base.
 */
const DEFAULTS = [
    "male",           // Default gender
    "body_standard",  // Default torso
    "head_standard",  // Default head
    "legs_pants"      // Default legs
];

/**
 * Fetches available tags from the server and initializes the UI toolbar.
 */
async function loadTags() {
    try {
        const response = await fetch('/tags');
        allTags = await response.json();
        const toolbar = document.getElementById('categories-toolbar');
        document.getElementById('loading-status').style.display = 'none';

        // Initialize grouped object
        const grouped = {};
        Object.keys(CATEGORIES).forEach(cat => grouped[cat] = []);
        
        // Group tags based on CATEGORIES keywords
        allTags.forEach((tag, index) => {
            let found = false;
            for (const [cat, keywords] of Object.entries(CATEGORIES)) {
                if (keywords.some(k => tag.includes(k))) {
                    grouped[cat].push({ tag, index });
                    found = true;
                    break;
                }
            }
            if (!found) grouped["OTHER"].push({ tag, index });
        });

        // Generate dropdown UI for each category
        Object.entries(grouped).forEach(([catName, tags]) => {
            if (tags.length === 0) return;

            const dropdown = document.createElement('div');
            dropdown.className = 'dropdown';
            dropdown.innerHTML = `
                <button class="drop-btn">${catName} ▾</button>
                <div class="dropdown-content">
                    <input type="text" class="search-mini" placeholder="Search..." oninput="filterTags(this)">
                    <div class="tags-list"></div>
                </div>
            `;

            const list = dropdown.querySelector('.tags-list');
            tags.forEach(item => {
                const btn = document.createElement('div');
                btn.className = 'tag-item';
                btn.setAttribute('data-index', item.index); 
                btn.textContent = item.tag.replace(/_/g, ' ');
                btn.onclick = () => toggleTag(item.index, btn, item.tag);
                list.appendChild(btn);
            });

            toolbar.appendChild(dropdown);
        });
    } catch (e) {
        console.error("Error loading tags:", e);
    }
}

/**
 * Toggles a tag selection. Includes exclusion logic for categories 
 * (e.g., selecting 'female' will deselect 'male').
 * 
 * @param {number} index - Index of the tag in allTags array.
 * @param {HTMLElement} element - The DOM element of the tag button.
 * @param {string} tagName - String identifier of the tag.
 */
function toggleTag(index, element, tagName) {
    const details = document.getElementById('selection-list-details');

    // 1. Identify category for exclusion logic
    const categoryKey = Object.keys(CATEGORIES).find(key => 
        CATEGORIES[key].some(keyword => tagName.includes(keyword))
    );

    // 2. Handle deselection
    if (selectedTags.has(index)) {
        selectedTags.delete(index);
        element.classList.remove('active');
        document.getElementById(`selected-${index}`)?.remove();
    } 
    else {
        // 3. EXCLUSION LOGIC: Remove existing selection from the same category
        if (categoryKey && categoryKey !== "OTHER") {
            selectedTags.forEach(selectedIdx => {
                const nameOfSelected = allTags[selectedIdx];
                const isSameCategory = CATEGORIES[categoryKey].some(k => nameOfSelected.includes(k));

                if (isSameCategory) {
                    selectedTags.delete(selectedIdx);
                    document.querySelector(`.tag-item[data-index="${selectedIdx}"]`)?.classList.remove('active');
                    document.getElementById(`selected-${selectedIdx}`)?.remove();
                }
            });
        }

        // 4. Finalize new selection
        selectedTags.add(index);
        element.classList.add('active');
        const item = document.createElement('div');
        item.id = `selected-${index}`;
        item.textContent = `• ${tagName}`;
        details.appendChild(item);
    }
    
    // Update UI counter
    document.getElementById('selection-count').textContent = selectedTags.size;
}

/**
 * Filters tags within a dropdown based on search input.
 * @param {HTMLInputElement} input - The search bar element.
 */
window.filterTags = (input) => {
    const filter = input.value.toLowerCase();
    const items = input.nextElementSibling.querySelectorAll('.tag-item');
    items.forEach(item => {
        item.style.display = item.textContent.toLowerCase().includes(filter) ? "" : "none";
    });
};

/**
 * Expands or collapses the list of currently selected traits.
 */
window.toggleSelectionList = () => {
    const list = document.getElementById('selection-list-details');
    const arrow = document.getElementById('arrow');
    list.classList.toggle('hidden');
    arrow.classList.toggle('rotate');
};

/**
 * Clears all current selections and resets the UI.
 */
function resetSelection() {
    selectedTags.clear();
    document.querySelectorAll('.tag-item').forEach(el => el.classList.remove('active'));
    document.getElementById('selection-list-details').innerHTML = '';
    document.getElementById('selection-count').textContent = '0';
}

document.getElementById('reset-btn').onclick = resetSelection;

/**
 * Randomizes character traits. Selects 3-6 unique tags across categories.
 */
document.getElementById('random-btn').onclick = () => {
    resetSelection();

    const countToSelect = Math.floor(Math.random() * 4) + 3;
    const chosenIndices = new Set();

    while(chosenIndices.size < countToSelect) {
        const randomIndex = Math.floor(Math.random() * allTags.length);
        chosenIndices.add(randomIndex);
    }

    chosenIndices.forEach(index => {
        const tagName = allTags[index];
        const btn = document.querySelector(`.tag-item[data-index="${index}"]`);
        
        if (btn) {
            toggleTag(index, btn, tagName);
        } else {
            selectedTags.add(index);
        }
    });
};

/**
 * Collects selected tags, applies defaults, and sends a request to the GAN model.
 */
document.getElementById('generate-btn').onclick = async () => {
    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.textContent = "FORGING...";

    try {
        let finalIndices = new Set(selectedTags);

        // Fill in missing core categories with DEFAULTS
        DEFAULTS.forEach(defaultTagName => {
            const defaultIndex = allTags.indexOf(defaultTagName);
            const categoryKey = Object.keys(CATEGORIES).find(key => 
                CATEGORIES[key].some(keyword => defaultTagName.includes(keyword))
            );
            
            const hasCategorySelected = Array.from(selectedTags).some(idx => {
                const name = allTags[idx];
                return CATEGORIES[categoryKey].some(k => name.includes(k));
            });

            if (!hasCategorySelected && defaultIndex !== -1) {
                finalIndices.add(defaultIndex);
            }
        });

        // Request generation from FastAPI backend
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tags: Array.from(finalIndices) })
        });

        const data = await response.json();
        if (data.image) {
            document.getElementById('output-image').src = `data:image/png;base64,${data.image}`;
        }
    } catch (err) {
        alert("Alchemical failure! Check console for details.");
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = "GENERATE";
    }
};

// Initial load
loadTags();