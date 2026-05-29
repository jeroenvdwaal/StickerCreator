# KDE HIG Reference

Apply the KDE Human Interface Guidelines when reviewing or writing UI code for this app.
Source: https://develop.kde.org/hig/

---

## 1. KDE app identity (`/hig/kde_app_design/`)
- Built with Qt + KDE Frameworks — mandatory
- GUI toolkit: **QtWidgets** for complex traditional desktop; **QtQuick + Kirigami** for modern/convergent (preferred for new apps)
- Philosophy: "Simple by default, powerful when needed"
- Target the 80% use case; don't limit to single-purpose simplicity
- Support customization for diverse workflows, not just aesthetics

---

## 2. Simple by default (`/hig/simple_by_default/`)
- Show actionable content immediately on launch — no blank states
- Use `Kirigami.PlaceholderMessage` for empty views: icon + explanation + action button
- Preselect last-used options; pre-populate fields when data available (e.g. clipboard)
- Enable inline editing instead of modal dialogs where possible
- Hide controls irrelevant to current context
- Progress indicator for tasks exceeding 1 second — `BusyIndicator` (indeterminate) or `LoadingPlaceholder` (determinate)
- Warn before destructive actions; move files to trash, never delete directly
- Offer undo for all content-removing actions; use `showPassiveNotification` for non-critical undos
- Use `Kirigami.InlineMessage` for truly destructive action warnings
- Always save window size/position (X11); remember active view, open files, scroll positions
- Don't minimize to system tray on close
- Size main window to display all labels without truncation

---

## 3. Powerful when needed (`/hig/powerful_when_needed/`)
- Implement keyboard shortcuts for every action; use `KStandardShortcut` for standard ones
- Shortcuts are accelerators — never the exclusive access method; primary workflows must stay visible
- Only two-finger gestures; three+ conflict with system gestures
- Reserve `Meta` key exclusively for global shortcuts
- No pages/sections labelled "Advanced" — use progressive disclosure (sub-pages, collapsed sections) instead
- Don't use settings to avoid design decisions or mask bugs
- Apply settings immediately — never require app restart
- Use `Kirigami.FormLayout` for configuration pages
- Expose Plasma-specific settings only on non-Plasma platforms

---

## 4. Layout and navigation (`/hig/layout_and_nav/`)
- Read order = layout order: top-leading = primary, bottom-trailing = final action
- Spacing with Kirigami units: title→content `smallSpacing`; group separation `largeSpacing`; window/page edges `largeSpacing`; rounded corners `cornerRadius`
- Fixed-size elements: `gridUnit` (18px) × N
- Icon sizes: menu/raised buttons `IconSizes.small`; flat/toolbar `IconSizes.smallMedium`; list items with subtitles `IconSizes.medium`
- Apps must adapt fluidly to resizing, maximizing, tiling
- Hamburger menu: static global actions only, disabled (not hidden) when irrelevant, ≤15 items; beyond 15 use full menubar
- Never include Quit/Minimize in hamburger menus
- Menubar: always visible, static, disabled when irrelevant
- Navigation: minimize as prerequisite; linear → `PageStack`; ≤5 destinations → `NavigationTabBar`; >5 → `GlobalDrawer`
- Launch view must always be first nav item
- RTL: test with `LANGUAGE=ar_AR [app_executable]`; use `-rtl` icon variants; don't mirror images

---

## 5. Displaying content (`/hig/displaying_content/`)
- Lists: textual content, fast scanning; grids: visual/wide content
- Implement items as Kirigami Delegates for automatic KDE styling
- Add controls on `Kirigami.InlineViewHeader`; remove controls inline on items, always visible
- Push a new Page when content fills most/all window area
- Use `Kirigami.Dialog` for user input; `OverlaySheet` for read-only narrow scrollable auxiliary content
- Add contrasting outline around overlaid elements (dark theme blending issue)
- Mutable tabs (documents): reorderable, span width, visible close buttons via `TabBar`
- Immutable tabs (settings): `Kirigami.NavigationTabBar`; hide tab bar when only one tab
- Tables: only for 3+ data pieces per item where cross-comparison matters; avoid 2-column tables (use list)
- Minimize tree views — confusing for typical users; prefer lists with collapsible sections
- Inline help: brief explanation below control; page-top for ≤2 sentence descriptions
- Never rely on hover tooltips for critical information (inaccessible to touch)
- Use `Kirigami.ContextualHelpButton` for lengthy explanations

---

## 6. Getting input (`/hig/getting_input/`)
- Button: general one-time actions; ToolButton: toolbar header/footer; RoundButton: floating over images (icon-only, unambiguous)
- Switch: instant-apply settings; CheckBox: settings requiring OK/Apply confirmation
- Don't change labels/icons when state changes; avoid checkable buttons
- RadioButton: ≤3 short options with vertical space; ComboBox: ≤10 items or limited vertical space; List view: >10 options
- Slider: speed > precision; SpinBox: precision paramount; Slider + SpinBox: both matter
- Text input only when automated validation impossible; validate input; show invalid state via `Kirigami.InlineMessage`; disable confirmation when invalid
- Use specialized Kirigami fields: `ActionTextField`, `SearchField`, `PasswordField`
- Dialogs: only for immediate required decisions or blocking progress; use `FileDialog` for file ops; `PromptDialog`/`MenuDialog`; never stack dialogs
- Pointing-finger cursor only for clickable URLs; underlined links only for external URLs

---

## 7. Communicating status (`/hig/status_changes/`)
- Minimize messages — only for long-running tasks users may have forgotten
- Show success visually (change related screen element) rather than sending confirmation messages
- Prevent errors first via structured inputs and validation
- Error handling priority: (1) make impossible/auto-recover → (2) describe + "Fix it" action → (3) describe + explain how to proceed → (4) description only
- Never use technical jargon in errors; never silent failures; all errors must be actionable
- Colors: Blue = selected/benign; Orange = warnings/non-default; Red = errors/dangerous — never color alone
- Passive notifications (low priority): `showPassiveNotification()`
- Attention-getting non-interrupting: `InlineMessage` with `Position.Header`
- System notifications: only when app backgrounded, actionable events, correct urgency level; never advertise features
- OSD: system-level only (volume/brightness) — not in windowed apps
- Task manager: show completion % for long-running tasks; unread counts for messaging
- System tray: last resort, opt-in only, show only during abnormal status when window hidden

---

## 8. Text and labels (`/hig/text_and_labels/`)
- Sentence case: labels ending with period/colon, tooltips, status messages, form labels
- Title case: window titles and proper nouns
- Oxford comma in lists; spaces around em-dashes
- Imperative mood for instructions; positive phrasing (describe enabled state)
- Avoid "you" — prefer impersonal: "Missing authorization" not "You are not authorized"
- Front-load important words; interactive labels max length ≈ "Configure Keyboard Shortcuts"
- Multi-sentence text: ≤85 chars/line (~450px)
- Ellipsis "…" (U+2026) only when additional user input required before action completes
- Window titles: distinctive, brief, describes content; no vendor/version; imperative verbs for dialogs
- Search fields: use `Kirigami.SearchField` (includes standard placeholder text)
- Implementation: `QtQuick.Controls.Label` for normal text, `Kirigami.Heading` for headers — avoid `QtQuick.Text` directly
- Use `i18n()` with KUIT semantic markup; `i18ncp()` for number-related text; allow 50%+ expansion for translations
- Replace violent/negative terms: Kill→Close, Execute→Run, Abort→Exit, Fatal→Critical, Slave→Worker
- App names: single memorable word related to purpose; don't force "K" prefix

---

## 9. Icons (`/hig/icons/`)
- Use system icon themes: `QIcon::fromTheme()` in QtWidgets; `icon.name` in QtQuick — no bundled custom icons/pixmaps
- Make `KIconThemes` a build dependency; call `KIconTheme::initTheme()` before `QApplication`
- Use `KStandardActions` for standard actions
- Sizes: 16×16 = symbolic only; 22×22 = almost always symbolic; 32×32+ = full-color
- Append `-symbolic` to request monochromatic
- Set icons on every button and menu item with text
- Don't duplicate same icon across multiple visible items
- Destructive: red trash (`edit-delete`) for user content; red X (`edit-delete-remove`) for removable abstract items; "Move to trash" uses black trash (`trash-empty`) with explicit label
- Icon-only buttons: only when space critical + icons are instantly recognizable (e.g. `list-add`, `configure`, `search`, `print`); when in doubt, show text
- Browse available icons: Icon Explorer in `plasma-sdk`; request missing at bugs.kde.org

---

## 10. Accessibility (`/hig/accessibility/`)
- Keyboard test: unplug mouse, interact with every element keyboard-only; focus must look visibly different from inactive selection
- All interactive elements must have accessible names via `Accessible` attached properties; no duplicate labels in same window
- Name items with distinctive parts first: "Germany (Europe)" not "Europe/Germany"
- Drag-and-drop: show item preview during drag; show "can't drag here" cursor on failed drop
- Virtual keyboard appears only when appropriate; hover tooltips also trigger on press-and-hold
- Don't rely on color alone — combine with icons, shapes, or text
- Test with system font size at 14pt; verify text scaling doesn't break layout
- Test with animations disabled; verify transitions work with instant/static fallback; no blinking elements except text cursors
- Test with Orca screen reader (display off)
- Don't assume physical abilities, age, gender, ethnicity, or technical skill level
- Use "system" not "phone"/"device"

---

## This app's HIG checklist
- [ ] `WelcomeWidget` = `PlaceholderMessage` equivalent: icon + label + open-file action
- [ ] Segmentation in `QThread` — UI never blocks; progress shown during model load
- [ ] Toast overlays for save/copy feedback (`showPassiveNotification` equivalent)
- [ ] `ShortcutsDialog` documents all keyboard shortcuts
- [ ] Toolbar icons via `QIcon::fromTheme()` with fallback; `KIconTheme::initTheme()` called
- [ ] Step indicator communicates current workflow stage
- [ ] Splitter: image viewer left, sticker preview right; proportions adjustable
- [ ] Destructive actions (clear image, reset) warn before executing
- [ ] Error messages: plain language, actionable, no jargon
- [ ] Window size/position persisted via `QSettings`
