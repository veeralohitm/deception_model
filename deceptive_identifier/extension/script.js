  // selecting loading div
const loader = document.querySelector("#loading");
const textOutput = document.querySelector("#showOutput");
var tablink;
// showing loading
function displayLoading() {
    loader.classList.add("display");
    // to stop loading after some time
    setTimeout(() => {
        loader.classList.remove("display");
    }, 30000);
}
chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
   tablink = tabs[0].url;
   console.log(tablink)
  // use `url` here inside the callback because it's asynchronous!
});
// hiding loading 
function hideLoading() {
    loader.classList.remove("display");
}
  async function getData() {
    displayLoading();
    let queryOptions = { active: true, lastFocusedWindow: true };
    let tab = await chrome.tabs.query(queryOptions);
    const url = tab[0].url;
    console.log(tablink);
    //const url = 'https://www.cnn.com/2022/12/03/asia/south-korea-worlds-lowest-fertility-rate-intl-hnk-dst';
    const options = {
        method: 'GET',
        headers: {
          'Access-Control-Allow-Origin': '*'
        },
      };
    const response = await fetch(`http://127.0.0.1:5000/api/summarize?url=${tablink}`,options);
    const data = await response.json();
    textOutput.innerText = data[0].summary_text;
    hideLoading();
  }

  getData();