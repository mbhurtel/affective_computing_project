
var music = document.getElementById("music-player")

var audio = {    
    init: function() {        
    var $that = this;        
        $(function() {            
            $that.components.media();        
        });    
    },
    components: {        
        media: function(target) {            
            var media = $('audio.fc-media', (target !== undefined) ? target : 'body');            
            if (media.length) {                
                media.mediaelementplayer({                    
                    audioHeight: 40,
                    features : ['playpause', 'current', 'duration', 'progress', 'volume', 'tracks', 'fullscreen', 'muted', 'autoplay'],
                    alwaysShowControls      : true,
                    timeAndDurationSeparator: '<span></span>',
                    iPadUseNativeControls: true,
                    iPhoneUseNativeControls: true,
                    AndroidUseNativeControls: true   ,
                    volume : 0.2             
                });            
            }        
        },
            
    },
};

audio.init();


// (setTimeout(playMusic, 1000));



// Webcam.set({
//     width: 960,
//     height: 660,
//     image_format: 'jpeg',
//     jpeg_quality: 90
// });
// Webcam.attach('#imageCapture');

// document.querySelector('#test').addEventListener('click', function () {
//     getExpression();
// });


// const getExpression = () => {
//     Webcam.snap(image_uri => {
//         console.log(image_uri)
//         // fetch('/expression', {
//         //     method: 'POST',
//         //     headers: {
//         //         'Accept': 'application/json',
//         //         'Content-Type': 'application/json'
//         //     },
//         //     body: JSON.stringify({ image_uri: image_uri })
//         // }).then(response => {
//         //     return response.json();
//         // }).then(res => {
//         //     mood = res.mood;
//         //     mood = mood.charAt(0).toUpperCase() + mood.slice(1);
//         //     document.querySelector('#status').innerHTML = `Current Mood: ${mood}`;
//         //     switch (mood) {
//         //         case "Angry":
//         //             playlist_index = 0;
//         //             audio.src = dir + angry_playlist[0] + ext;
//         //             current_song.innerHTML = angry_title[playlist_index];
//         //             $("#circle-image img").attr("src", angry_poster[playlist_index]);
//         //             $("body").css("background-image", "linear-gradient(to bottom, rgb(255, 0, 0) , rgb(255, 0, 76))");
//         //             break;
//         //         case "Happy":
//         //             playlist_index = 0;
//         //             audio.src = dir + happy_playlist[0] + ext;
//         //             current_song.innerHTML = happy_title[playlist_index];
//         //             $("#circle-image img").attr("src", happy_poster[playlist_index]);
//         //             $("body").css("background-image", "linear-gradient(to bottom, rgba(188, 203, 7, 1) 0%, rgba(219, 203, 88, 1) 100%)");
//         //             break;
//         //         case "Calm":
//         //             playlist_index = 0;
//         //             audio.src = dir + calm_playlist[0] + ext;
//         //             current_song.innerHTML = calm_title[playlist_index];
//         //             $("#circle-image img").attr("src", calm_poster[playlist_index]);
//         //             $("body").css("background-image", "linear-gradient(to bottom, rgba(137, 170, 75, 1) 0%, rgba(77, 138, 9, 1) 100%)");
//         //             break;
//         //         case "Sad":
//         //             playlist_index = 0;
//         //             audio.src = dir + sad_playlist[0] + ext;
//         //             current_song.innerHTML = sad_title[playlist_index];
//         //             $("#circle-image img").attr("src", sad_poster[playlist_index]);
//         //             $("body").css("background-image", "linear-gradient(to bottom, rgba(14, 9, 121, 1) 69%, rgba(0, 189, 255, 1) 100%)");
//         //             break;
//         //     }
//         // });
//     });
// }

// setTimeout(() => { getExpression() }, 2000);


// songs = data.songs
// console.log("Songs Received: ", songs)
// if(songs.length > 0){
//     console.log("Creating Content for songs")
//     var content= ""
//     songs.forEach((song) => {
//         content += createContent(song)
//     })
//     let music_container = document.getElementById("music-player-container")
//     print("Music Container ", music_container)
//     music_container.innerHTML = content
//     console.log("Content Updated")
// }