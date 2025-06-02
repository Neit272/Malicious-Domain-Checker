async function fetchData(domain, tabId) {
  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ domain: domain }),
  });

  const record = await res.json();

  document.getElementById("checking").style.display = "none";

  if (window._mal_timer) clearTimeout(window._mal_timer);
  if (window._mal_interval) clearInterval(window._mal_interval);

  if (record.label === "Malicious") {
    document.getElementById("malicious").style.display = "block";
    document.getElementById("benign").style.display = "none";
    document.getElementById("prob-mal").textContent = `Confidence: ${Math.round(
      record.probability
    )}%`;

    let secondsLeft = 5;
    let timerSpan = document.getElementById("mal-timer");
    if (!timerSpan) {
      timerSpan = document.createElement("span");
      timerSpan.id = "mal-timer";
      timerSpan.style.display = "block";
      timerSpan.style.marginTop = "8px";
      timerSpan.style.fontWeight = "bold";
      document.getElementById("malicious").appendChild(timerSpan);
    }
    timerSpan.textContent = `This page will be unloaded in ${secondsLeft} seconds`;

    window._mal_interval = setInterval(() => {
      secondsLeft--;
      if (secondsLeft > 0) {
        timerSpan.textContent = `This page will be unloaded in ${secondsLeft} seconds`;
      } else {
        clearInterval(window._mal_interval);
      }
    }, 1000);

    window._mal_timer = setTimeout(() => {
      document.body.innerHTML =
        '<div style="height:100vh;display:flex;align-items:center;justify-content:center;font-size:2rem;">Session terminated</div>';
      if (tabId) chrome.tabs.remove(tabId);
    }, 5000);
  } else {
    document.getElementById("benign").style.display = "block";
    document.getElementById("malicious").style.display = "none";
    document.getElementById("prob-benign").textContent = `Confidence: ${100 - Math.round(record.probability)}%`;}}

chrome.tabs.query({ active: true, lastFocusedWindow: true }, function (tabs) {
  const url = new URL(tabs[0].url);
  const domain = url.hostname;
  document.getElementById("checking").style.display = "block";
  fetchData(domain, tabs[0].id);
});
