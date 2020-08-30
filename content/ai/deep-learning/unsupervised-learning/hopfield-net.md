---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 320

# Basic metadata
title: "Hopfield Nets"
date: 2020-08-18
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "Unsupervised Learning"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

share: false  # Show social sharing links?
featured: true

comments: false  # Show comments?
disable_comment: true
commentable: false  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.
editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Optional header image (relative to `static/img/` folder).
header:
  caption: ""
  image: ""

# Menu
menu: 
    deep-learning:
        parent: unsupervised-learning
        weight: 2

---

## **Binary Hopfield Nets**

### Basic Structure: Binary Unit

- Single layer of processing units

- Each unit $i$ has an activity value or “state” $u\_i$

  - Binary: $-1$ or $1$
  - Denoted as $+$ and $–$ respectively

- Example

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2016.52.56.png" alt="截屏2020-08-18 16.52.56" style="zoom: 67%;" />

### Connections

- Processing units fully interconnected

- Weights from unit $j$ to unit $i$: $T\_{ij}$

- No unit has a connection with itself
  $$
  \forall i :  \qquad T\_{ii} = 0  
  $$

- Weights between a pair of units are **symmetric**
  $$
  T\_{ji} = T\_{ij}
  $$

  - Symmetric weights lead to the fact that the network will converge (relax in stable state)

- Example

  <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="501px" viewBox="-0.5 -0.5 501 403" content="&lt;mxfile host=&quot;app.diagrams.net&quot; modified=&quot;2020-08-18T20:20:59.064Z&quot; agent=&quot;5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36&quot; etag=&quot;icQIK4C8tDW4pi8ZyByC&quot; version=&quot;13.6.2&quot; type=&quot;device&quot;&gt;&lt;diagram id=&quot;5oDFmgM-eGvQfvP3TnwN&quot; name=&quot;Page-1&quot;&gt;7Zldb5swFIZ/DdJ20QlsyMdl89H1ots0Zdqaq8oCB7waHDkmgf36mWAgfGXtlgmS5iqc18bY57HfYykanPrRR47W3ifmYKoB3Yk0ONMAMEwwkD+JEqfKcGikgsuJozoVwoL8wkrUlRoSB29KHQVjVJB1WbRZEGBblDTEOduVu60YLX91jVxcExY2onX1B3GEl6oAwnHRcI+J66lPQ6irmfso663G2HjIYbtU2veBcw1OOWMiffKjKaZJ9rLEpAPdtbTmM+M4EC954WHMzOD569P8G7K/hPefw5nn3VhqbiLOVowdmQAVMi485rIA0XmhTjgLAwcno+oyKvo8MLaWoiHFn1iIWNFEoWBS8oRPVaucMI8f1fv7YJkEH6wsnEWHjbNYRSsWCDUoSFud24SxjAMW4FS5I5Sq/hvB2XPODUolXW2yxNYkZrRYyG18JHPZbkTcxeJIP5CjlocEMx/LFcn3OKZIkG15HkjtVjfvV/CUDwrpK/AOusVbEF2WgDbj/RPOKv5e4IVd4h3X8P47vIiIx4Pn5cFzwS0J4kOI9dN8ibDNLmGrcbeIhupLmjV99/1J6vo+lvY/MaT2vn7oKZUFM0nyziMCL9Zon46drNll+mizTqvoikTJLnoxhi3mAkfHQdQTp16AA1Uz1SXByOJdUXINVd1076Da5uLJkz26GudfniVwFmdJP8pXZfFkZpp55ivM1Lj4DdBp5QQtZgpyM705Sysdgt5Z6bAbK+3RiYDnYImw5UTAMz8R1cuFaXZ+IsyWVJtnf5Or2k8Pkm0035tbNrNcuignMs3TlFHGCzNZSSepSIgSN5ChLVOFpT5JEklsRG9Vg08ch7aRKxteldUJyFjVO/aoTmbUAMb8b1yaS/Bb4wKtMhcIuubSXAhaHeliyVh6mQwYdk2muW68PTID2Dcy1tXLkuo/7JuXDa5cGmp/7myn5yLD4h+tfdvBH4Nw/hs=&lt;/diagram&gt;&lt;/mxfile&gt;" onclick="(function(svg){var src=window.event.target||window.event.srcElement;while (src!=null&amp;&amp;src.nodeName.toLowerCase()!='a'){src=src.parentNode;}if(src==null){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://viewer.diagrams.net/?client=1&amp;edit=_blank');}}})(this);" style="cursor:pointer;max-width:100%;max-height:403px;"><defs><style xmlns="http://www.w3.org/1999/xhtml" type="text/css">.MathJax_Preview {color: #888}&#xa;#MathJax_Message {position: fixed; left: 1em; bottom: 1.5em; background-color: #E6E6E6; border: 1px solid #959595; margin: 0px; padding: 2px 8px; z-index: 102; color: black; font-size: 80%; width: auto; white-space: nowrap}&#xa;#MathJax_MSIE_Frame {position: absolute; top: 0; left: 0; width: 0px; z-index: 101; border: 0px; margin: 0px; padding: 0px}&#xa;.MathJax_Error {color: #CC0000; font-style: italic}&#xa;</style><style xmlns="http://www.w3.org/1999/xhtml" type="text/css">.MathJax_Hover_Frame {border-radius: .25em; -webkit-border-radius: .25em; -moz-border-radius: .25em; -khtml-border-radius: .25em; box-shadow: 0px 0px 15px #83A; -webkit-box-shadow: 0px 0px 15px #83A; -moz-box-shadow: 0px 0px 15px #83A; -khtml-box-shadow: 0px 0px 15px #83A; border: 1px solid #A6D ! important; display: inline-block; position: absolute}&#xa;.MathJax_Menu_Button .MathJax_Hover_Arrow {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 4px; -webkit-border-radius: 4px; -moz-border-radius: 4px; -khtml-border-radius: 4px; font-family: 'Courier New',Courier; font-size: 9px; color: #F0F0F0}&#xa;.MathJax_Menu_Button .MathJax_Hover_Arrow span {display: block; background-color: #AAA; border: 1px solid; border-radius: 3px; line-height: 0; padding: 4px}&#xa;.MathJax_Hover_Arrow:hover {color: white!important; border: 2px solid #CCC!important}&#xa;.MathJax_Hover_Arrow:hover span {background-color: #CCC!important}&#xa;</style><style xmlns="http://www.w3.org/1999/xhtml" type="text/css">.MathJax_SVG_Display {text-align: center; margin: 1em 0em; position: relative; display: block!important; text-indent: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; width: 100%}&#xa;.MathJax_SVG .MJX-monospace {font-family: monospace}&#xa;.MathJax_SVG .MJX-sans-serif {font-family: sans-serif}&#xa;#MathJax_SVG_Tooltip {background-color: InfoBackground; color: InfoText; border: 1px solid black; box-shadow: 2px 2px 5px #AAAAAA; -webkit-box-shadow: 2px 2px 5px #AAAAAA; -moz-box-shadow: 2px 2px 5px #AAAAAA; -khtml-box-shadow: 2px 2px 5px #AAAAAA; padding: 3px 4px; z-index: 401; position: absolute; left: 0; top: 0; width: auto; height: auto; display: none}&#xa;.MathJax_SVG {display: inline; font-style: normal; font-weight: normal; line-height: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; padding: 0; margin: 0}&#xa;.MathJax_SVG * {transition: none; -webkit-transition: none; -moz-transition: none; -ms-transition: none; -o-transition: none}&#xa;.MathJax_SVG &gt; div {display: inline-block}&#xa;.mjx-svg-href {fill: blue; stroke: blue}&#xa;.MathJax_SVG_Processing {visibility: hidden; position: absolute; top: 0; left: 0; width: 0; height: 0; overflow: hidden; display: block!important}&#xa;.MathJax_SVG_Processed {display: none!important}&#xa;.MathJax_SVG_test {font-style: normal; font-weight: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-transform: none; letter-spacing: normal; word-spacing: normal; overflow: hidden; height: 1px}&#xa;.MathJax_SVG_test.mjx-test-display {display: table!important}&#xa;.MathJax_SVG_test.mjx-test-inline {display: inline!important; margin-right: -1px}&#xa;.MathJax_SVG_test.mjx-test-default {display: block!important; clear: both}&#xa;.MathJax_SVG_ex_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60ex}&#xa;.mjx-test-inline .MathJax_SVG_left_box {display: inline-block; width: 0; float: left}&#xa;.mjx-test-inline .MathJax_SVG_right_box {display: inline-block; width: 0; float: right}&#xa;.mjx-test-display .MathJax_SVG_right_box {display: table-cell!important; width: 10000em!important; min-width: 0; max-width: none; padding: 0; border: 0; margin: 0}&#xa;.MathJax_SVG .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}&#xa;</style></defs><g><path d="M 130 61 L 370 61" fill="none" stroke="#000000" stroke-width="3" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 70 121 L 70 281" fill="none" stroke="#000000" stroke-width="3" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 112.43 103.43 L 387.57 298.57" fill="none" stroke="#000000" stroke-width="3" stroke-miterlimit="10" pointer-events="stroke"/><ellipse cx="70" cy="61" rx="60" ry="60" fill="#ffffff" stroke="#000000" stroke-width="3" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 61px; margin-left: 11px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-16-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="8.479ex" height="2.21ex" viewBox="0 -743.6 3650.5 951.6" role="img" focusable="false" style="vertical-align: -0.483ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M52 648Q52 670 65 683H76Q118 680 181 680Q299 680 320 683H330Q336 677 336 674T334 656Q329 641 325 637H304Q282 635 274 635Q245 630 242 620Q242 618 271 369T301 118L374 235Q447 352 520 471T595 594Q599 601 599 609Q599 633 555 637Q537 637 537 648Q537 649 539 661Q542 675 545 679T558 683Q560 683 570 683T604 682T668 681Q737 681 755 683H762Q769 676 769 672Q769 655 760 640Q757 637 743 637Q730 636 719 635T698 630T682 623T670 615T660 608T652 599T645 592L452 282Q272 -9 266 -16Q263 -18 259 -21L241 -22H234Q216 -22 216 -15Q213 -9 177 305Q139 623 138 626Q133 637 76 637H59Q52 642 52 648Z"/><g transform="translate(583,-150)"><path stroke-width="1" transform="scale(0.707)" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g><g transform="translate(1315,0)"><path stroke-width="1" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"/></g><g transform="translate(2371,0)"><path stroke-width="1" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z"/></g><g transform="translate(3149,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-16">V_1 = +1</script></div></div></div></foreignObject><text x="70" y="67" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">V_1 = +1</text></switch></g><path d="M 430 121 L 430 281" fill="none" stroke="#000000" stroke-width="3" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 387.57 103.43 L 112.43 298.57" fill="none" stroke="#000000" stroke-width="3" stroke-miterlimit="10" pointer-events="stroke"/><ellipse cx="430" cy="61" rx="60" ry="60" fill="#ffffff" stroke="#000000" stroke-width="3" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 61px; margin-left: 371px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-4-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="8.479ex" height="2.21ex" viewBox="0 -743.6 3650.5 951.6" role="img" focusable="false" style="vertical-align: -0.483ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M52 648Q52 670 65 683H76Q118 680 181 680Q299 680 320 683H330Q336 677 336 674T334 656Q329 641 325 637H304Q282 635 274 635Q245 630 242 620Q242 618 271 369T301 118L374 235Q447 352 520 471T595 594Q599 601 599 609Q599 633 555 637Q537 637 537 648Q537 649 539 661Q542 675 545 679T558 683Q560 683 570 683T604 682T668 681Q737 681 755 683H762Q769 676 769 672Q769 655 760 640Q757 637 743 637Q730 636 719 635T698 630T682 623T670 615T660 608T652 599T645 592L452 282Q272 -9 266 -16Q263 -18 259 -21L241 -22H234Q216 -22 216 -15Q213 -9 177 305Q139 623 138 626Q133 637 76 637H59Q52 642 52 648Z"/><g transform="translate(583,-150)"><path stroke-width="1" transform="scale(0.707)" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z"/></g><g transform="translate(1315,0)"><path stroke-width="1" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"/></g><g transform="translate(2371,0)"><path stroke-width="1" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"/></g><g transform="translate(3149,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-4">V_2 = -1</script></div></div></div></foreignObject><text x="430" y="67" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">V_2 = -1</text></switch></g><path d="M 130 341 L 370 341" fill="none" stroke="#000000" stroke-width="3" stroke-miterlimit="10" pointer-events="stroke"/><ellipse cx="70" cy="341" rx="60" ry="60" fill="#ffffff" stroke="#000000" stroke-width="3" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 341px; margin-left: 11px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-8-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="8.479ex" height="2.306ex" viewBox="0 -743.6 3650.5 992.8" role="img" focusable="false" style="vertical-align: -0.579ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M52 648Q52 670 65 683H76Q118 680 181 680Q299 680 320 683H330Q336 677 336 674T334 656Q329 641 325 637H304Q282 635 274 635Q245 630 242 620Q242 618 271 369T301 118L374 235Q447 352 520 471T595 594Q599 601 599 609Q599 633 555 637Q537 637 537 648Q537 649 539 661Q542 675 545 679T558 683Q560 683 570 683T604 682T668 681Q737 681 755 683H762Q769 676 769 672Q769 655 760 640Q757 637 743 637Q730 636 719 635T698 630T682 623T670 615T660 608T652 599T645 592L452 282Q272 -9 266 -16Q263 -18 259 -21L241 -22H234Q216 -22 216 -15Q213 -9 177 305Q139 623 138 626Q133 637 76 637H59Q52 642 52 648Z"/><g transform="translate(583,-150)"><path stroke-width="1" transform="scale(0.707)" d="M127 463Q100 463 85 480T69 524Q69 579 117 622T233 665Q268 665 277 664Q351 652 390 611T430 522Q430 470 396 421T302 350L299 348Q299 347 308 345T337 336T375 315Q457 262 457 175Q457 96 395 37T238 -22Q158 -22 100 21T42 130Q42 158 60 175T105 193Q133 193 151 175T169 130Q169 119 166 110T159 94T148 82T136 74T126 70T118 67L114 66Q165 21 238 21Q293 21 321 74Q338 107 338 175V195Q338 290 274 322Q259 328 213 329L171 330L168 332Q166 335 166 348Q166 366 174 366Q202 366 232 371Q266 376 294 413T322 525V533Q322 590 287 612Q265 626 240 626Q208 626 181 615T143 592T132 580H135Q138 579 143 578T153 573T165 566T175 555T183 540T186 520Q186 498 172 481T127 463Z"/></g><g transform="translate(1315,0)"><path stroke-width="1" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"/></g><g transform="translate(2371,0)"><path stroke-width="1" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"/></g><g transform="translate(3149,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-8">V_3 = -1</script></div></div></div></foreignObject><text x="70" y="347" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">V_3 = -1</text></switch></g><ellipse cx="430" cy="341" rx="60" ry="60" fill="#ffffff" stroke="#000000" stroke-width="3" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 341px; margin-left: 371px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-17-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="8.479ex" height="2.21ex" viewBox="0 -743.6 3650.5 951.6" role="img" focusable="false" style="vertical-align: -0.483ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M52 648Q52 670 65 683H76Q118 680 181 680Q299 680 320 683H330Q336 677 336 674T334 656Q329 641 325 637H304Q282 635 274 635Q245 630 242 620Q242 618 271 369T301 118L374 235Q447 352 520 471T595 594Q599 601 599 609Q599 633 555 637Q537 637 537 648Q537 649 539 661Q542 675 545 679T558 683Q560 683 570 683T604 682T668 681Q737 681 755 683H762Q769 676 769 672Q769 655 760 640Q757 637 743 637Q730 636 719 635T698 630T682 623T670 615T660 608T652 599T645 592L452 282Q272 -9 266 -16Q263 -18 259 -21L241 -22H234Q216 -22 216 -15Q213 -9 177 305Q139 623 138 626Q133 637 76 637H59Q52 642 52 648Z"/><g transform="translate(583,-150)"><path stroke-width="1" transform="scale(0.707)" d="M462 0Q444 3 333 3Q217 3 199 0H190V46H221Q241 46 248 46T265 48T279 53T286 61Q287 63 287 115V165H28V211L179 442Q332 674 334 675Q336 677 355 677H373L379 671V211H471V165H379V114Q379 73 379 66T385 54Q393 47 442 46H471V0H462ZM293 211V545L74 212L183 211H293Z"/></g><g transform="translate(1315,0)"><path stroke-width="1" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"/></g><g transform="translate(2371,0)"><path stroke-width="1" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z"/></g><g transform="translate(3149,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-17">V_4 = +1</script></div></div></div></foreignObject><text x="430" y="347" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">V_4 = +1</text></switch></g><rect x="210" y="21" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 41px; margin-left: 211px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-7-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2.971ex" height="2.115ex" viewBox="0 -743.6 1279 910.4" role="img" focusable="false" style="vertical-align: -0.387ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"/><g transform="translate(778,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-7">-1</script></div></div></div></foreignObject><text x="250" y="47" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">-1</text></switch></g><rect x="0" y="161" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 181px; margin-left: 1px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-10-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2.971ex" height="2.115ex" viewBox="0 -743.6 1279 910.4" role="img" focusable="false" style="vertical-align: -0.387ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"/><g transform="translate(778,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-10">-1</script></div></div></div></foreignObject><text x="40" y="187" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">-1</text></switch></g><rect x="150" y="111" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 131px; margin-left: 151px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-12-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2.971ex" height="2.115ex" viewBox="0 -743.6 1279 910.4" role="img" focusable="false" style="vertical-align: -0.387ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z"/><g transform="translate(778,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-12">+1</script></div></div></div></foreignObject><text x="190" y="137" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">+1</text></switch></g><rect x="280" y="111" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 131px; margin-left: 281px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-13-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2.971ex" height="2.115ex" viewBox="0 -743.6 1279 910.4" role="img" focusable="false" style="vertical-align: -0.387ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M56 237T56 250T70 270H369V420L370 570Q380 583 389 583Q402 583 409 568V270H707Q722 262 722 250T707 230H409V-68Q401 -82 391 -82H389H387Q375 -82 369 -68V230H70Q56 237 56 250Z"/><g transform="translate(778,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-13">+1</script></div></div></div></foreignObject><text x="320" y="137" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">+1</text></switch></g><rect x="420" y="161" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 181px; margin-left: 421px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-14-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2.971ex" height="2.115ex" viewBox="0 -743.6 1279 910.4" role="img" focusable="false" style="vertical-align: -0.387ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"/><g transform="translate(778,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-14">-1</script></div></div></div></foreignObject><text x="460" y="187" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">-1</text></switch></g><rect x="210" y="341" width="80" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 361px; margin-left: 211px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span class="MathJax_Preview" style=""></span><span class="MathJax_SVG" id="MathJax-Element-15-Frame" tabindex="0" style="font-size: 100%; display: inline-block;"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2.971ex" height="2.115ex" viewBox="0 -743.6 1279 910.4" role="img" focusable="false" style="vertical-align: -0.387ex;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><path stroke-width="1" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"/><g transform="translate(778,0)"><path stroke-width="1" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"/></g></g></svg></span><script type="math/tex" id="MathJax-Element-15">-1</script></div></div></div></foreignObject><text x="250" y="367" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">-1</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://desk.draw.io/support/solutions/articles/16000042487" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Viewer does not support full SVG 1.1</text></a></switch></svg>

Unit vector:
$$
U = (+1, -1, -1, +1)^T
$$
Weight matrix:


$$
T=\left(\begin{array}{cccc}
T\_{11} & T\_{12} & T\_{13} & T\_{14} \\\\
T\_{21} & T\_{22} & T\_{23} & T\_{24} \\\\
T\_{31} & T\_{32} & T\_{33} & T\_{34} \\\\
T\_{41} & T\_{42} & T\_{43} & T\_{44}
\end{array}\right)
= \left(\begin{array}{cccc}
0 & -1 & -1 & +1 \\\\
-1 & 0 & +1 & -1 \\\\
-1 & +1 & 0 & -1 \\\\
+1 & -1 & -1 & 0
\end{array}\right)
$$

### Update Binary Unit

$$
u\_i = \operatorname{sign}(\sum\_{j} T\_{ji} u\_j) = \begin{cases}
+1 & \text{if }\sum\_{j} T\_{ji} u\_j \geq 0 \\\\
-1 & \text {otherwise }
\end{cases}
$$

1. Evaluate the sum of the weighted inputs
2. Set state $1$ if the sum is greater or equal $0$, else $-1$

### Update Procedure

- Network state is initialized in the beginning
- Update
  - **Asynchronous**: Update one unit at a time
  - **Synchronous**: Update all nodes in parallel

- Continue updating until the network state does not change anymore

#### Example

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.19.17.png" alt="截屏2020-08-18 17.19.17" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.23.26.png" alt="截屏2020-08-18 17.23.26" style="zoom:67%;" />



> $$
> u\_4 = \operatorname{sign}(+1 \cdot (-1) + (-1) \cdot 1 + (-1) \cdot 1) = \operatorname{sign}(-3) = -1
> $$
>
> So the new state of unit 4 is $-$



<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.23.29.png" alt="截屏2020-08-18 17.23.29" style="zoom:67%;" />



<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.23.32.png" alt="截屏2020-08-18 17.23.32" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.23.34.png" alt="截屏2020-08-18 17.23.34" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18 17.23.37.png" alt="截屏2020-08-18 17.23.37" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.23.39.png" alt="截屏2020-08-18 17.23.39" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2017.23.42.png" alt="截屏2020-08-18 17.23.42" style="zoom:67%;" />

#### Order of updating

- Could be sequentially
- Random order (Hopfield networks)
  - Same average update rate

  - Advantages in implementation

  - Advantages in function (equiprobable stable states)

- **Randomized asynchronous** updating is a closer match to the biological neuronal nets

### Energy function

- Assign a numerical value to each possible state of the system (**Lyapunov Function**)

- Corresponds to the “energy” of the net
  $$
  \begin{aligned}
  E &= -\frac{1}{2} \sum\_{j} \sum\_{i \neq j} u\_{i} T\_{j i} u\_{j}  \\\\
  &= -\frac{1}{2}U^T TU
  \end{aligned}
  $$

#### Proof on Convergence

**Each updating step leads to lower or same energy in the net.**

Let's say only unit $j$ is updated at a time. Energy changes only for unit $j$ is
$$
E\_{j}=-\frac{1}{2} \sum\_{i \neq j} u\_{i}T\_{j i} u\_{j}
$$
Given a change in state, the difference in Energy $E$ is
$$
\begin{aligned}
\Delta E\_{j}&=E\_{j\_{n e w}}-E\_{j\_{o l d}} \\\\
&=-\frac{1}{2} \Delta u\_{j} \sum\_{i \neq j} u\_{j} T\_{j i}
\end{aligned}
$$

$$
\Delta u\_{j}=u\_{j\_{n e w}}-u\_{j\_{o l d}}
$$

- Change from $-1$ to $1$:
  $$
  \Delta u\_{j}=2, \Sigma T\_{j i} u\_{i} \geq 0 \Rightarrow \Delta E\_{j} \leq 0
  $$

- Change from $1$ to $-1$:
  $$
  \Delta u\_{j}=-2, \Sigma T\_{j i} u\_{i}<0 \Rightarrow \Delta E\_{j}<0
  $$

#### Stable States

- Stable states are minima of the energy function
  - Can be global or local minima

- Analogous to finding a minimum in a mountainous terrain

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2022.36.03.png" alt="截屏2020-08-18 22.36.03" style="zoom: 67%;" />

## Applications

### Associative memory

### Optimization



## Limitations

### Found stable state (memory) is not guaranteed the most similar pattern to the input pattern

Not all memories are remembered with same emphasis (attractors region is not the same size)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2022.39.28.png" alt="截屏2020-08-18 22.39.28" style="zoom: 67%;" />

### Spurious States

- Retrieval States 
- Reversed States

- Mixture States: Any linear combination of an odd number of patterns

- “Spinglass” states: Stable states that are no linear combination of stored patterns (occur when too many patterns are stored)

### Efficiency

- In a net of $N$ units, patterns of length $N$ can be stored

- Assuming uncorrelated patterns, the capacity $C$ of a hopfield net is
  $$
  C \approx 0.15N
  $$

  - Tighter bound
    $$
    \frac{N}{4 \ln N}<C<\frac{N}{2 \ln N}
    $$

## Reference

- [Hopfield Networks are useless. Here’s why you should learn them.](https://towardsdatascience.com/hopfield-networks-are-useless-heres-why-you-should-learn-them-f0930ebeadcd)
- [Working with a Hopfield neural network model](https://www.youtube.com/watch?v=HoWJzeAT9uc)

