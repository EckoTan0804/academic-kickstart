---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 150

# Basic metadata
title: "Learn PyTorch with Example"
date: 2020-09-10
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "PyTorch"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: ""
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
    pytorch:
        parent: getting-started
        weight: 5
---

## TL;DR

- PyTorch provides two main features:
  - An n-dimensional Tensor, similar to numpy but can run on GPUs
  - Automatic differentiation for building and training neural networks

- Typical procedure of neural network training with PyTorch

  1. Define network structure

     - Use `torch.nn.Sequential`, e.g.: 

       ```python
       model = torch.nn.Sequential(
           torch.nn.Linear(D_in, H),
           torch.nn.ReLU(),
           torch.nn.Linear(H, D_out),
       )
       ```

       or

     - Define own Modules by

       - subclassing `nn.Module`
       - defining a `forward` function which receives input Tensors

       ```python
       import torch
       
       class TwoLayerNet(torch.nn.Module):
           
           def __init__(self, D_in, H, D_out):
               """
               In the constructor we instantiate two nn.Linear modules and assign them as
               member variables.
               """
               super(TwoLayerNet, self).__init__()
               self.linear1 = torch.nn.Linear(D_in, H)
               self.linear2 = torch.nn.Linear(H, D_out)
       
           def forward(self, x):
               """
               In the forward function we accept a Tensor of input data and we must 
               return a Tensor of output data. 
               We can use Modules defined in the constructor as well as arbitrary 
               operators on Tensors.
               """
               h_relu = self.linear1(x).clamp(min=0)
               y_pred = self.linear2(h_relu)
               return y_pred
       ```

  2. Define loss function and optimizer (and learning rate)

     - Loss function: implemented in [`torch.nn`](https://pytorch.org/docs/stable/nn.html#loss-functions)

       - E.g.: Mean Square Loss

         ```python
         loss_fn = torch.nn.MSELoss(reduction='sum')
         ```

     - Optimizer (see: [`torch.optim`](https://pytorch.org/docs/stable/optim.html)) and learning rate

       - E.g.: Adam

         ```python
         learning_rate = 1e-4
         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
         ```

   3.  Iterate training dataset multiple times. In each iteration

        	1. Forward pass
        	2. Compute loss
        	3. Zero all of the parameters' gradients 
        	4. Backward pass
        	5. Update parameters

       ```python
       for t in range(500):
           # 3.1 Forward pass
           y_pred = model(x)
       
           # 3.2 Compute and print loss
           loss = loss_fn(y_pred, y)
           if t % 100 == 99:
               print(t, loss.item())
       
       		# 3.3 Zero gradients
           optimizer.zero_grad()
       
           # 3.4 Backward pass
           loss.backward()
       
           # 3.5 Update parameters
           optimizer.step()
       ```

### Diagramm Summary

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="2487px" viewBox="-0.5 -0.5 2487 1941" content="&lt;mxfile host=&quot;app.diagrams.net&quot; modified=&quot;2020-09-11T13:59:49.511Z&quot; agent=&quot;5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36&quot; etag=&quot;Kx3auAzAgyNpoAYbRczD&quot; version=&quot;13.6.9&quot; type=&quot;device&quot;&gt;&lt;diagram id=&quot;ZzJhBbZltDj1jY3X3ib4&quot; name=&quot;structure&quot;&gt;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&lt;/diagram&gt;&lt;/mxfile&gt;" onclick="(function(svg){var src=window.event.target||window.event.srcElement;while (src!=null&amp;&amp;src.nodeName.toLowerCase()!='a'){src=src.parentNode;}if(src==null){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://viewer.diagrams.net/?client=1&amp;edit=_blank');}}})(this);" style="cursor:pointer;max-width:100%;max-height:1941px;"><defs/><g><path d="M 436 1190 L 707.76 1190" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 713.76 1190 L 705.76 1194 L 707.76 1190 L 705.76 1186 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="316" y="950" width="120" height="480" fill="#dae8fc" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="376" cy="1000" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="376" cy="1100" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="376" cy="1380" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><path d="M 836 1190 L 1107.76 1190" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1113.76 1190 L 1105.76 1194 L 1107.76 1190 L 1105.76 1186 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="716" y="810" width="120" height="760" fill="#d5e8d4" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="776" cy="860" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="776" cy="960" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="776" cy="1510" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><path d="M 1236 1190 L 1499.63 1190" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1504.88 1190 L 1497.88 1193.5 L 1499.63 1190 L 1497.88 1186.5 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="1116" y="910" width="120" height="560" fill="#fff2cc" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="1176" cy="970" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="1176" cy="1070" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><ellipse cx="1176" cy="1410" rx="40" ry="40" fill="none" stroke="#000000" stroke-width="2" pointer-events="all"/><rect x="476" y="1130" width="200" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 198px; height: 1px; padding-top: 1150px; margin-left: 477px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Linear + ReLU</div></div></div></foreignObject><text x="576" y="1159" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">Linear + ReLU</text></switch></g><rect x="876" y="1130" width="200" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 198px; height: 1px; padding-top: 1150px; margin-left: 877px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Linear</div></div></div></foreignObject><text x="976" y="1159" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">Linear</text></switch></g><rect x="296" y="610" width="150" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 148px; height: 1px; padding-top: 650px; margin-left: 297px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Input<br />size: <font face="Courier New">D_in</font></div></div></div></foreignObject><text x="371" y="659" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">Input...</text></switch></g><rect x="701" y="610" width="150" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 148px; height: 1px; padding-top: 650px; margin-left: 702px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Hidden<br />size: <font face="Courier New">H</font></div></div></div></foreignObject><text x="776" y="659" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">Hidden...</text></switch></g><rect x="1096" y="610" width="170" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 168px; height: 1px; padding-top: 650px; margin-left: 1097px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Output<br />size: <font face="Courier New">D_out</font></div></div></div></foreignObject><text x="1181" y="659" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">Output...</text></switch></g><rect x="256" y="0" width="1210" height="550" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe flex-start; justify-content: unsafe flex-start; width: 1208px; height: 1px; padding-top: 7px; margin-left: 258px;"><div style="box-sizing: border-box; font-size: 0; text-align: left; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><div><b><font face="Helvetica">0. Define network structure, loss function, and optimizer</font></b></div><div><br /></div><div>import torch</div><div><br /></div><div>model = torch.nn.Sequential(</div><div>    torch.nn.Linear(D_in, H),</div><div>    torch.nn.ReLU(),</div><div>    torch.nn.Linear(H, D_out),</div><div>)</div><div><br /></div><div>loss_fn = torch.nn.MSELoss(reduction='sum')<br /></div><div><br /></div><div><div>learning_rate = 1e-4</div><div>optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)</div></div></div></div></div></foreignObject><text x="258" y="37" fill="#000000" font-family="Courier New" font-size="30px">0. Define network structure, loss function, and optimizer...</text></switch></g><path d="M 1666 1190 L 1887.76 1190" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1893.76 1190 L 1885.76 1194 L 1887.76 1190 L 1885.76 1186 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="1506" y="1160" width="160" height="60" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 158px; height: 1px; padding-top: 1190px; margin-left: 1507px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">y_pred</div></div></div></foreignObject><text x="1586" y="1199" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">y_pred</text></switch></g><path d="M 1976 1220 L 1976 1331.76" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1976 1337.76 L 1972 1329.76 L 1976 1331.76 L 1980 1329.76 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="1896" y="1160" width="160" height="60" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 158px; height: 1px; padding-top: 1190px; margin-left: 1897px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">y</div></div></div></foreignObject><text x="1976" y="1199" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">y</text></switch></g><rect x="1876" y="1010" width="200" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 198px; height: 1px; padding-top: 1030px; margin-left: 1877px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">ground truth</div></div></div></foreignObject><text x="1976" y="1039" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">ground truth</text></switch></g><rect x="586" y="1600" width="360" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe flex-start; justify-content: unsafe flex-start; width: 358px; height: 1px; padding-top: 1607px; margin-left: 588px;"><div style="box-sizing: border-box; font-size: 0; text-align: left; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font face="Helvetica"><b>1. Forward</b></font><br />y_pred = model(x)</div></div></div></foreignObject><text x="588" y="1637" fill="#000000" font-family="Courier New" font-size="30px">1. Forward...</text></switch></g><path d="M 216 1730 L 216 1700 Q 216 1690 226 1690 L 1576 1690 Q 1586 1690 1586 1680 L 1586 1228.24" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1586 1222.24 L 1590 1230.24 L 1586 1228.24 L 1582 1230.24 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="56" y="1730" width="320" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 318px; height: 1px; padding-top: 1770px; margin-left: 57px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 40px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; font-weight: bold; white-space: normal; word-wrap: normal; ">model.parameters()</div></div></div></foreignObject><text x="216" y="1782" fill="#000000" font-family="Courier New" font-size="40px" text-anchor="middle" font-weight="bold">model.parameters...</text></switch></g><rect x="2016" y="1210" width="470" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 468px; height: 1px; padding-top: 1250px; margin-left: 2018px;"><div style="box-sizing: border-box; font-size: 0; text-align: left; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><b><font face="Helvetica">2. Compute loss</font></b><br />loss = loss_fn(y_pred, y)</div></div></div></foreignObject><text x="2018" y="1259" fill="#000000" font-family="Courier New" font-size="30px">2. Compute loss...</text></switch></g><path d="M 1976 1400 L 1976 1830 Q 1976 1840 1966 1840 L 954.24 1840" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 948.24 1840 L 956.24 1836 L 954.24 1840 L 956.24 1844 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="1876" y="1350" width="200" height="50" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 198px; height: 1px; padding-top: 1375px; margin-left: 1877px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">loss</div></div></div></foreignObject><text x="1976" y="1384" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">loss</text></switch></g><path d="M 746 1840 L 626 1840 Q 616 1840 606 1840 L 226 1840 Q 216 1840 216 1830 L 216 1818.24" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 216 1812.24 L 220 1820.24 L 216 1818.24 L 212 1820.24 Z" fill="#000000" stroke="#000000" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><rect x="746" y="1810" width="200" height="60" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 198px; height: 1px; padding-top: 1840px; margin-left: 747px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">gradient</div></div></div></foreignObject><text x="846" y="1849" fill="#000000" font-family="Helvetica" font-size="30px" text-anchor="middle">gradient</text></switch></g><rect x="2016" y="1530" width="400" height="120" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 398px; height: 1px; padding-top: 1590px; margin-left: 2017px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><div style="text-align: left"><b><font face="Helvetica">3. Backward</font></b></div>optimizer.zero_grad()<br /><div style="text-align: left"><span>loss.backward()</span></div></div></div></div></foreignObject><text x="2216" y="1599" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">3. Backwardoptimizer.zero_g...</text></switch></g><rect x="296" y="1850" width="360" height="90" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe flex-start; justify-content: unsafe flex-start; width: 358px; height: 1px; padding-top: 1857px; margin-left: 298px;"><div style="box-sizing: border-box; font-size: 0; text-align: left; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font face="Helvetica"><b>4. Update parameters</b></font> <br />optimizer.step()</div></div></div></foreignObject><text x="298" y="1887" fill="#000000" font-family="Courier New" font-size="30px">4. Update parameters...</text></switch></g><path d="M 376 1250 L 376 1180" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" stroke-dasharray="2 6" pointer-events="stroke"/><path d="M 775 1100 L 775 1030" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" stroke-dasharray="2 6" pointer-events="stroke"/><path d="M 1175 1210 L 1175 1140" fill="none" stroke="#000000" stroke-width="2" stroke-miterlimit="10" stroke-dasharray="2 6" pointer-events="stroke"/></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://desk.draw.io/support/solutions/articles/16000042487" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Viewer does not support full SVG 1.1</text></a></switch></svg>

## From `numpy` to `pytorch`

View in [nbviewer](https://nbviewer.jupyter.org/github/EckoTan0804/summay-pytorch/blob/master/pytorch-quick-start/05-learn-pytorch-with-examples.ipynb)

## Reference

- [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#)