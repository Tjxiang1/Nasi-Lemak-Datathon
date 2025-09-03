/*
document.getElementById("submit").addEventListener("click", async () => {
    const keyword = document.getElementById("keyword").value;
    document.getElementById("search").innerText = "(From javascript) Searching the keyword #" + keyword;

    try{
        const response = await fetch("/search?keyword=" + keyword);
        const data = await response.json();
        
        document.getElementById("output").innerText = data.message;
    }
    catch(error){
        document.getElementById("search").innerText = "Error searching";
    }

})

*/