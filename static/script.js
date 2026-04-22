let selectedTags = new Set();
let allTags = [];

// Słownik kategorii - dopasuj nazwy do swoich tagów w JSON
const CATEGORIES = {
    "SEX": ["male", "female"],
    "HEAD": ["head_", "hair_"],
    "BODY": ["body_", "chest_", "torso_"],
    "ARMS": ["arms_", "hands_"],
    "LEGS": ["legs_", "feet_"],
    "WEAPON": ["weapon_", "sword_", "staff_", "bow_"],
    "SHIELD": ["shield_"],
    "INNE": [] // Reszta trafi tutaj
};

// Tagi, które zostaną użyte, jeśli użytkownik nie wybierze nic z danej kategorii
const DEFAULTS = [
    "male",           // Domyślna płeć
    "body_standard",  // Domyślny korpus
    "head_standard",  // Domyślna głowa
    "legs_pants"      // Domyślne nogi
];

async function loadTags() {
    try {
        const response = await fetch('/tags');
        allTags = await response.json();
        const toolbar = document.getElementById('categories-toolbar');
        document.getElementById('loading-status').style.display = 'none';

        // Grupowanie tagów
        const grouped = {};
        Object.keys(CATEGORIES).forEach(cat => grouped[cat] = []);
        
        allTags.forEach((tag, index) => {
            let found = false;
            for (const [cat, keywords] of Object.entries(CATEGORIES)) {
                if (keywords.some(k => tag.includes(k))) {
                    grouped[cat].push({ tag, index });
                    found = true;
                    break;
                }
            }
            if (!found) grouped["INNE"].push({ tag, index });
        });

        // Tworzenie menu dropdown dla każdej kategorii
        Object.entries(grouped).forEach(([catName, tags]) => {
            if (tags.length === 0) return;

            const dropdown = document.createElement('div');
            dropdown.className = 'dropdown';
            dropdown.innerHTML = `
                <button class="drop-btn">${catName} ▾</button>
                <div class="dropdown-content">
                    <input type="text" class="search-mini" placeholder="Szukaj..." oninput="filterTags(this)">
                    <div class="tags-list"></div>
                </div>
            `;

            const list = dropdown.querySelector('.tags-list');
            tags.forEach(item => {
                const btn = document.createElement('div');
                btn.className = 'tag-item';
                // Dodajemy atrybut, aby łatwo znaleźć przycisk podczas losowania
                btn.setAttribute('data-index', item.index); 
                btn.textContent = item.tag.replace(/_/g, ' ');
                btn.onclick = () => toggleTag(item.index, btn, item.tag);
                list.appendChild(btn);
            });

            toolbar.appendChild(dropdown);
        });
    } catch (e) {
        console.error("Błąd ładowania tagów:", e);
    }
}

function toggleTag(index, element, tagName) {
    const details = document.getElementById('selection-list-details');

    // 1. Znajdź kategorię, do której należy ten tag
    const categoryKey = Object.keys(CATEGORIES).find(key => 
        CATEGORIES[key].some(keyword => tagName.includes(keyword))
    );

    // 2. Jeśli tag jest już wybrany - po prostu go usuń (odznaczanie)
    if (selectedTags.has(index)) {
        selectedTags.delete(index);
        element.classList.remove('active');
        document.getElementById(`selected-${index}`)?.remove();
    } 
    else {
        // 3. LOGIKA WYKLUCZANIA: Jeśli to kategoria inna niż "INNE"
        if (categoryKey && categoryKey !== "INNE") {
            // Szukamy wszystkich już wybranych tagów z TEJ SAMEJ kategorii
            selectedTags.forEach(selectedIdx => {
                const nameOfSelected = allTags[selectedIdx];
                const isSameCategory = CATEGORIES[categoryKey].some(k => nameOfSelected.includes(k));

                if (isSameCategory) {
                    // Usuwamy poprzedni wybór z tej kategorii
                    selectedTags.delete(selectedIdx);
                    // Usuwamy podświetlenie z przycisku w HTML
                    document.querySelector(`.tag-item[data-index="${selectedIdx}"]`)?.classList.remove('active');
                    // Usuwamy z listy tekstowej
                    document.getElementById(`selected-${selectedIdx}`)?.remove();
                }
            });
        }

        // 4. Dodaj nowy wybór
        selectedTags.add(index);
        element.classList.add('active');
        const item = document.createElement('div');
        item.id = `selected-${index}`;
        item.textContent = `• ${tagName}`;
        details.appendChild(item);
    }
    
    // Aktualizacja licznika
    document.getElementById('selection-count').textContent = selectedTags.size;
}

// Funkcja wyszukiwania wewnątrz kategorii
window.filterTags = (input) => {
    const filter = input.value.toLowerCase();
    const items = input.nextElementSibling.querySelectorAll('.tag-item');
    items.forEach(item => {
        item.style.display = item.textContent.toLowerCase().includes(filter) ? "" : "none";
    });
};

// Funkcja rozwijania listy wybranych
window.toggleSelectionList = () => {
    const list = document.getElementById('selection-list-details');
    const arrow = document.getElementById('arrow');
    list.classList.toggle('hidden');
    arrow.classList.toggle('rotate');
};

// --- FUNKCJA RESETU ---
function resetSelection() {
    selectedTags.clear();
    // Czyścimy podświetlenia przycisków
    document.querySelectorAll('.tag-item').forEach(el => el.classList.remove('active'));
    // Czyścimy listę tekstową
    document.getElementById('selection-list-details').innerHTML = '';
    // Zerujemy licznik
    document.getElementById('selection-count').textContent = '0';
}

document.getElementById('reset-btn').onclick = resetSelection;

// --- FUNKCJA RANDOM ---
document.getElementById('random-btn').onclick = () => {
    resetSelection();

    // Losujemy od 3 do 6 unikalnych tagów
    const countToSelect = Math.floor(Math.random() * 4) + 3;
    const chosenIndices = new Set();

    while(chosenIndices.size < countToSelect) {
        const randomIndex = Math.floor(Math.random() * allTags.length);
        chosenIndices.add(randomIndex);
    }

    // Aktywujemy wylosowane tagi
    chosenIndices.forEach(index => {
        const tagName = allTags[index];
        // Szukamy przycisku w DOM po atrybucie data-index
        const btn = document.querySelector(`.tag-item[data-index="${index}"]`);
        
        // Jeśli przycisk istnieje w menu, używamy toggleTag, by zaktualizować UI
        if (btn) {
            toggleTag(index, btn, tagName);
        } else {
            // Jeśli tag nie jest w żadnej kategorii (mało prawdopodobne), dodajemy go tylko do Setu
            selectedTags.add(index);
        }
    });
};

// Przycisk GENERATE
document.getElementById('generate-btn').onclick = async () => {
    const btn = document.getElementById('generate-btn');
    btn.disabled = true;
    btn.textContent = "WYKUWANIE...";

    try {
        // 1. Tworzymy kopię wybranych indeksów
        let finalIndices = new Set(selectedTags);

        // 2. Sprawdzamy, czy wybrano kluczowe elementy (Płeć, Głowa, Ciało, Nogi)
        // Jeśli nie, dodajemy domyślne indeksy
        DEFAULTS.forEach(defaultTagName => {
            const defaultIndex = allTags.indexOf(defaultTagName);
            
            // Szukamy, czy użytkownik wybrał już cokolwiek z tej kategorii
            // (np. jeśli default to 'male', sprawdzamy czy wybrano płeć)
            const categoryKey = Object.keys(CATEGORIES).find(key => 
                CATEGORIES[key].some(keyword => defaultTagName.includes(keyword))
            );
            
            const hasCategorySelected = Array.from(selectedTags).some(idx => {
                const name = allTags[idx];
                return CATEGORIES[categoryKey].some(k => name.includes(k));
            });

            // Jeśli kategoria jest pusta i tag domyślny istnieje w bazie - dodaj go
            if (!hasCategorySelected && defaultIndex !== -1) {
                finalIndices.add(defaultIndex);
            }
        });

        // 3. Wysyłamy wzbogaconą listę tagów
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
        alert("Błąd alchemiczny!");
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = "GENERATE";
    }
};

loadTags();