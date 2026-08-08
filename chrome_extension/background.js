// background.js - Service worker for handling sidebar toggle
chrome.action.onClicked.addListener(async (tab) => {
  // Open the side panel - sidebar handles YouTube detection internally
  await chrome.sidePanel.open({ windowId: tab.windowId });
});