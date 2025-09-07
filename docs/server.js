const express = require('express');
const app = express();
const axios = require('axios')

app.listen(3000);

app.set('view engine', 'ejs');
app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.render('index');
})

app
    .route('/search')
    .get(async (req, res) => {

    })
    .post(async (req, res) => {
        const keyword = req.body.keyword;

        try{
            const response = await axios.get('http://localhost:5000/generateGraph', {
                params: {keyword}
            })
            info = response.data;
            //console.log(info)
            res.render('index', info)
        } 
        catch(error){
            res.status(500).send("Error connecting with Python API");
        }

    })


app.post('/compare', async (req, res) =>{
    keywords = []
    keywords.push(req.body.keyword);

    try{
        const response = await axios.get('http://localhost:5000/compareGraph?' + 
            `keywords=${encodeURIComponent(keywords.join(','))}`
        )
        //console.log(response.data);
        res.render('index', { imagePaths: response.data , keyWords: keywords})
    } 
    catch(error){
        res.status(500).send("Error connecting with Python API");
    }    
})