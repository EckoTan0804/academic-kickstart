---
# Basic info
title: "zip"
date: 2020-07-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Python", "Basics"]
categories: ["Coding"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: ""
share: false  # Show social sharing links?
featured: true
lastmod: true

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
    python:
        parent: py-basics
        weight: 2
---


`zip()`: creates an iterator that will aggregate elements from two or more iterables. 

According to the [official documentation](https://docs.python.org/3/library/functions.html#zip), Python’s `zip()` function behaves as follows:

> Returns an iterator of tuples, where the *i*-th tuple contains the *i*-th element from each of the argument sequences or iterables. The iterator stops when the shortest input iterable is exhausted. With a single iterable argument, it returns an iterator of 1-tuples. With no arguments, it returns an empty iterator.

Python zip operations work just like the physical zipper on a bag or pair of jeans. Interlocking pairs of teeth on both sides of the zipper are pulled together to close an opening.

## Use `zip()` in python

`zip(*iterables)`:

- takes in [iterables](https://docs.python.org/3/glossary.html#term-iterable) as arguments and returns an **iterator**

- generates a series of tuples containing elements from each iterable

- can accept any type of iterable, such as [files](https://realpython.com/read-write-files-python/), [lists, tuples](https://realpython.com/python-lists-tuples/), [dictionaries](https://realpython.com/python-dicts/), [sets](https://realpython.com/python-sets/), and so on.

Pass n arguments


```python
# Pass n arguments
numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
upper_letters = ['A', 'B', 'C']
zipped = zip(numbers, letters, upper_letters)
```


```python
zipped
```




    <zip at 0x62479ae08>




```python
list(zipped)
```




    [(1, 'a', 'A'), (2, 'b', 'B'), (3, 'c', 'C')]




```python
num_tuple = (1, 2)
lettet_tuple = ('a', 'b')
upper_letter_tuple = ('A', 'B')
list(zip(num_tuple, lettet_tuple, upper_letter_tuple))
```




    [(1, 'a', 'A'), (2, 'b', 'B')]



Pass no arguments


```python
# Passing no argument
zipped = zip()
```


```python
zipped
```




    <zip at 0x62473c908>




```python
list(zipped)
```




    []



Pass one arguments


```python
# Pass one argument
zipped = zip(numbers)
list(zipped)
```




    [(1,), (2,), (3,)]



Pass arguments of **unequal** length: 

the number of elements that `zip()` puts out will be equal to the length of the **shortest** iterable. The remaining elements in any longer iterables will be totally ignored by zip()


```python
list(zip(range(5), range(100)))
```




    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]



If trailing or unmatched values are important, then use [`itertools.zip_longest()`](https://docs.python.org/3/library/itertools.html#itertools.zip_longest) instead of `zip()`


```python
from itertools import zip_longest

longest = range(5)
zipped = zip_longest(numbers, letters, longest, fillvalue='?')
```


```python
list(zipped)
```




    [(1, 'a', 0), (2, 'b', 1), (3, 'c', 2), ('?', '?', 3), ('?', '?', 4)]




```python

```

## Loop Over Multiple Iterables

Looping over multiple iterables is one of the most common use cases for Python’s `zip()` function

### Traverse lists in parallel


```python
letters
```




    ['a', 'b', 'c']




```python
numbers
```




    [1, 2, 3]




```python
for l,n in zip(letters, numbers):
    print(f'letter: {l}')
    print(f'number: {n} \n')
```

    letter: a
    number: 1 
    
    letter: b
    number: 2 
    
    letter: c
    number: 3 



### Traverse dictionaries in parallel


```python
dict_one = {'name': 'John', 'last_name': 'Doe', 'job': 'Python Consultant'}
dict_two = {'name': 'Jane', 'last_name': 'Doe', 'job': 'Community Manager'}

for (k1, v1), (k2, v2) in zip(dict_one.items(), dict_two.items()):
    print(k1, '->', v1)
    print(k2, '->', v2, '\n')
```

    name -> John
    name -> Jane 
    
    last_name -> Doe
    last_name -> Doe 
    
    job -> Python Consultant
    job -> Community Manager 



### Unzip a sequence

`zip(*zipped)`


```python
numbers
```




    [1, 2, 3]




```python
letters
```




    ['a', 'b', 'c']




```python
zipped = zip(numbers, letters)
```


```python
zipped_list = list(zipped)
zipped_list
```




    [(1, 'a'), (2, 'b'), (3, 'c')]



We have a list of tuples and want to separate the elements of each tuple into independent sequences. To do this, we can use `zip()` along with the [unpacking operator `*`](https://realpython.com/python-kwargs-and-args/#unpacking-with-the-asterisk-operators),


```python
list(zip(*zipped_list))
```




    [(1, 2, 3), ('a', 'b', 'c')]

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="3043px" viewBox="-0.5 -0.5 3043 2062" content="&lt;mxfile host=&quot;app.diagrams.net&quot; modified=&quot;2020-07-06T16:28:01.174Z&quot; agent=&quot;5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36&quot; etag=&quot;1l50Gr0cZYkznZ8h5aUm&quot; version=&quot;13.3.9&quot; type=&quot;device&quot;&gt;&lt;diagram id=&quot;GIBQfwL42elWJfeV2FHo&quot; name=&quot;Page-1&quot;&gt;7Z1tk6JIEoB/jTEzG9EdFO98VHt7+uJmZmdvYnfv7ssGDajsIjhYduv8+q3iTSHLFhQsQPpLS6HQnU+SmZWZVY6k6XL7MTRXi8+B7XgjUbC3I+lhJIpIFlXyi47s4hFNQ/HAPHTt5E37gW/uDycZFJLRjWs769wbcRB42F3lB63A9x0L58bMMAxe82+bBV7+ritz7oCBb5bpwdE/XBsv4lFJEoT9iSfHnS+SW5NTRnxmaabvTt66Xph28HowJP08kqZhEOD41XI7dTwqvVQw8ecej5zN/rLQ8XGZD1je0tkan8zti/DLvx/FP3ba4+QOSWJ8nRfT2yT/c/Ln4l0qhDDY+LZDLyOMpMnrwsXOt5Vp0bOvBDsZW+ClR44QeTlzPW8aeEFIjv3Ad+hQ4ONHc+l6VAmmwSZ0nZDc4YvzmpxMuEtCcpx+fiQSQdMfMm6b60X0N9CbrHEY/J0RkcgIlEYioBcnxM72YCiRzkcnWDo43JG3JGf1BFSiqmp6/LoHj5SU5uIAupEOmom2zbNr73mQFwmSSnjkmvGcEOMpek+O9+Jg1zLz5NS3yNWABqlqHg5CAoOOrDLoIFFoDA+A42+WazISHQgjZULeQS4tiPEvaaQ8AHxEBDjPKMZSxADJmJ4798mhRQRLnihpQgVK2Hjj5MTStW16G6ZS5NXmAKYonHpm6yAq5oGKMuSZjR3ilJuCCQ2h52Ai1xzPd+N3Mcp3k/TF9F27qZaxvGJNz6lSgipxnlekKgGqP9zVyrH/9Nw1zsjSg/fkxPv4+aVcE/gfPnQcrVQTWiNPVlchWY1lfpt7XtXTrnFOZLYq/89ncaT5nF5BeFMoRSsmiSWtmN6YVDQgFSiWEzGCuV7FsfTM3VJ9q2ZGDvWehAGz2Uy0LPCQkDO2+qwqKlM73wZeOaI7oKFfFYYOYEg3CwMxLMZ1aRiABro9GrykX2Ym07S5VtS2mWtJAVIZc9VJ23T0GVMnVUt3nmfldTID3mKdhCHE5Pak3xJnKcHIZXqzMLg7Sxk6SwDD8e0xzbGSI8sz12vXKqQCS2SOENvLJfbdsXPp2Tcth8IQTzoWOp6J3Ze812fJLLnD18D18cHcRytknwS1IPc1US3LST4mHiRhC1dCglC4FCpeCpvh3MHgUhHF7D8/H2wqkhxY1aMz2vXK9HOE1e8bmp2OeN3NEpJjOmsWVXO5ishJRLmliZU9Wj55tIpn91cir+bJby+98vFbriMdoTeUhNUWXuWn3Fw+viIRSnzR/I3IcPzvpcPDpD4yM4WkNxK1kiFRY/N6FWZVjwSKFc37US7Hw+9jAXuDMaoh5YnojEQ3yxM0CATw6EtGQS3/wOzykufllmGM1JeEQnUW3GMkOHfrSz6hAgxewkcSFDYNF78lh4l3zvvXIMSLYB74pvcpCFYJg78cjHeJWM0NDvKEnK2L/0s/fq8pyeH/kqvR1w/bw4PdOX68fN30TVZxCHradsTx5en3nYy9SwfVl3kfkXs4kE33jk0QGwwHULHEwD8egEntqCrai5l6pm3tNXxwYh7Vom9N/C0JyBCcTtOOgFvFwT0m02DB/4i7aHAKh/Ime9+BdNgGweoky3oj6o9VYfmlL7O4DHnptEexmYxRBrruRALOJPoyrbsYDquF6LpwYP6jL9O88+Fwg9Hn4LM6DsQ7/FH6HI1ewKMtnkXtc3h6OR/uzkWF4ep7QKdbtamKawEuJZuVZAHIqzbyqDC+HkCeBVLiTRIG4wPJc0hm3RfcSMLIvevd/FcmmZwtLo3jbmxhGXQAWwtY7sYXzu8GsnWQ5W+MWQ2F1ZrAZGYTWLRc60jv16A41adIckFxOOuNBmewg950QG+4u5J0p4lBcbqlONw9lVZipdLNtL7v+9Mv730HJuLave8anNqPlBt7aqtviwKrpeUeUNTYvhsaY+l35/dgaHwPFbHQWY5kRvb7yiDhRD63pKPTQCs2d1YGqhaXEjGXswoMoI21Bmpw+t7ZxUV0O5DDBUYfhhVG51gdQ9bvlbyeaqxdYVhlt8b01Kg7xKvdDLQl/AO9UpJybvhnKEATwMUaDgANRgDIhTvBHe7iZQWqpKQDdGHBnXAvyGo6sl9eEB3tDo++OqFLBEItwsObunSy419qlc6JhWmCXGyKLKtxYmHZroyurG0ldii6trbpolrQNiTot6xtxRWwcnEzx7LapomctQ3G0ty1LWfZhBNqdr5GpWFnS1SK+DXg6ZRiLFPaiDEupl1btUpsBtH4VmuGBuSADEYgiRhgm+ur16BooGy62R2cUe9sX3164R721V8Mh3vrow7rM33pqz8fDjcYIoDRn7766ji499XrsC+4P331F/BojWeBCaT+9NVfzoe/cxmaeC8k25JWTx3mUAaQZ4Hk3Y+jw/zEQPIcktwbZPTjVb6hs6oNitLSVk59aAHupN5wdx1pmm9QnG4pDndPZcAc07Cc5ByyxVIq72mBAfNVA9hawHK39TDzNZCtgyx3Y5y1kfIsI4oSLCMq/MuISICeCgqnm9WQPffOFhKRAP1NXyqJl+Phnu1FAnQafaklXoCHHw5GdaQ31cQzgHAvJyIB1kP6U0+8hEh7PAwsdPSnolgDoRY4maGAcSnblmQPaMvzgLIelLzzBUiAVYWB5Vks+WcIGLupD4n+VqlKS0uLCA27fXRTc/g7EARzO4PqdEF1WuCvhmJFTWzbVmBECOawBrT1oOVv8WE2bGBbC9sWmOQSi64bX6tYXMKuMcQioawQmasxNrbZThnJ1LVA+MjeZm8j4718t/TS3N0RVA0vw0XM3ZLobZIo8zlMA0wrWD67xBh1wKgd2R1PrGK4MtWukIJUwBOJWDX/4g4ltT2Ocok9aNYLc0VfukuTPhqT6Pc4zfFTCYGEf4rlk/nseF+DtYvdgOJ5DjAOlgxumH4RKqN+SX5Koooe9uRPfFhgvFpHsyKilo+Ob4W7FXbsO/zsC/fzNSZPpnVP1JOcjD5C3vv4nf4Zz/5IVKTx+IttfLR+3T1pM2nz8PnpN7z9/dN/lH89/TX7Okc/Jl9+//UjGpvIFqbLlTj9bkj/X/ziLrVsX6jN+vuKGqfxb5He0RtGArijFYVGVCqz+nmjLzM0TGbafGSI94bemEcssTFE4203GirZdnNtp4hYixla4RXLfg/vrXtFBAsAg1vMP/kdcouZFR38Yjv9YnmdSs8Wt+FqiV8UWSUDpl+s2EZR/pvFs2awY+1jTbpk2AlLt8IEbK775eIohdDHRtgK22fu8sLn1vQiwgx3bxpfq+Pg/vXWSISzxt50ulbgwU/+JXYQbXwypYIt4VphuRkdjP3p+s3At1k54Uy/R12+FQi0xnsyvq66P0291YG0wH+W2Mpy2Pk7chRwoZxx9k7MMkj/iUJxo92m80TpHz+gP4VeMWpErzKmmVdHD2cxUYqwlWlA21wvooOyXyTV+DcA5fM3KsOtZlm761hxCUbhG38gWj4lpxZX2Ga9H4dQa3LN5DAMaGfk/pkmUlp8DmyHvuMf&lt;/diagram&gt;&lt;/mxfile&gt;" onclick="(function(svg){var src=window.event.target||window.event.srcElement;while (src!=null&amp;&amp;src.nodeName.toLowerCase()!='a'){src=src.parentNode;}if(src==null){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://app.diagrams.net/?client=1&amp;lightbox=1&amp;edit=_blank');}}})(this);" style="cursor:pointer;max-width:100%;max-height:2062px;"><defs/><g><rect x="1" y="440" width="1520" height="920" fill="none" stroke="#000000" stroke-width="3" stroke-dasharray="9 9" pointer-events="all"/><rect x="1581" y="860" width="1460" height="1200" fill="none" stroke="#000000" stroke-width="3" stroke-dasharray="9 9" pointer-events="all"/><rect x="41" y="0" width="240" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 238px; height: 1px; padding-top: 20px; margin-left: 42px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">nums = [1, 2, 3]</div></div></div></foreignObject><text x="161" y="26" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">nums = [1, 2, 3]</text></switch></g><rect x="441" y="0" width="330" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 328px; height: 1px; padding-top: 20px; margin-left: 442px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">letters = ['A', 'B', 'C']</div></div></div></foreignObject><text x="606" y="26" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">letters = ['A', 'B', 'C']</text></switch></g><rect x="11" y="620" width="760" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 758px; height: 1px; padding-top: 640px; margin-left: 12px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">zipped_list = list(zip(nums, letters))</div></div></div></foreignObject><text x="391" y="649" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">zipped_list = list(zip(nums, letters))</text></switch></g><rect x="121" y="80" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 120px; margin-left: 122px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">2</div></div></div></foreignObject><text x="161" y="126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">2</text></switch></g><rect x="201" y="80" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 120px; margin-left: 202px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">3</div></div></div></foreignObject><text x="241" y="126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">3</text></switch></g><rect x="41" y="80" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 120px; margin-left: 42px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">1</div></div></div></foreignObject><text x="81" y="126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">1</text></switch></g><rect x="481" y="80" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 120px; margin-left: 482px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">A</div></div></div></foreignObject><text x="521" y="126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">A</text></switch></g><rect x="561" y="80" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 120px; margin-left: 562px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">B</div></div></div></foreignObject><text x="601" y="126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">B</text></switch></g><rect x="641" y="80" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 120px; margin-left: 642px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">C</div></div></div></foreignObject><text x="681" y="126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">C</text></switch></g><path d="M 891 820 L 919.17 913.9" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 920.68 918.93 L 915.31 913.23 L 919.17 913.9 L 922.02 911.22 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="1601" y="1030" width="240" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 238px; height: 1px; padding-top: 1050px; margin-left: 1602px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span style="font-family: &quot;courier new&quot;"><font style="font-size: 30px">*zipped_list</font></span></div></div></div></foreignObject><text x="1721" y="1059" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">*zipped_list</text></switch></g><rect x="851" y="560" width="80" height="240" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><rect x="851" y="640" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 680px; margin-left: 852px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">2</div></div></div></foreignObject><text x="891" y="686" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">2</text></switch></g><rect x="851" y="720" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 760px; margin-left: 852px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">3</div></div></div></foreignObject><text x="891" y="766" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">3</text></switch></g><rect x="851" y="560" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 600px; margin-left: 852px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">1</div></div></div></foreignObject><text x="891" y="606" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">1</text></switch></g><path d="M 911 720 L 911 720" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 911 720 L 911 720 L 911 720 L 911 720 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="1041" y="560" width="80" height="240" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><rect x="1041" y="560" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 600px; margin-left: 1042px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'A'</div></div></div></foreignObject><text x="1081" y="606" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'A'</text></switch></g><rect x="1041" y="640" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 680px; margin-left: 1042px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'B'</div></div></div></foreignObject><text x="1081" y="686" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'B'</text></switch></g><rect x="1041" y="720" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 760px; margin-left: 1042px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'C'</div></div></div></foreignObject><text x="1081" y="766" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'C'</text></switch></g><rect x="891" y="1080" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1120px; margin-left: 892px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">2</div></div></div></foreignObject><text x="931" y="1126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">2</text></switch></g><rect x="891" y="1200" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1240px; margin-left: 892px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">3</div></div></div></foreignObject><text x="931" y="1246" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">3</text></switch></g><rect x="891" y="960" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1000px; margin-left: 892px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">1</div></div></div></foreignObject><text x="931" y="1006" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">1</text></switch></g><rect x="1011" y="960" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1000px; margin-left: 1012px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'A'</div></div></div></foreignObject><text x="1051" y="1006" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'A'</text></switch></g><rect x="1011" y="1080" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1120px; margin-left: 1012px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'B'</div></div></div></foreignObject><text x="1051" y="1126" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'B'</text></switch></g><rect x="1011" y="1200" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1240px; margin-left: 1012px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'C'</div></div></div></foreignObject><text x="1051" y="1246" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'C'</text></switch></g><rect x="831" y="970" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1010px; margin-left: 832px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="851" y="1028" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="831" y="1090" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1130px; margin-left: 832px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="851" y="1148" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="831" y="1210" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1250px; margin-left: 832px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="851" y="1268" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="1111" y="970" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1010px; margin-left: 1112px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="1131" y="1028" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="1111" y="1090" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1130px; margin-left: 1112px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="1131" y="1148" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="1111" y="1210" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1250px; margin-left: 1112px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="1131" y="1268" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="971" y="980" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1020px; margin-left: 972px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="991" y="1038" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="971" y="1090" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1130px; margin-left: 972px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="991" y="1148" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="971" y="1210" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1250px; margin-left: 972px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="991" y="1268" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><path d="M 1081 820 L 1062.25 913.76" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1061.22 918.9 L 1059.16 911.35 L 1062.25 913.76 L 1066.02 912.73 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="781" y="960" width="40" height="100" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1010px; margin-left: 782px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">[</div></div></div></foreignObject><text x="801" y="1028" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">[</text></switch></g><rect x="1151" y="1200" width="40" height="100" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1250px; margin-left: 1152px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">]</div></div></div></foreignObject><text x="1171" y="1268" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">]</text></switch></g><rect x="521" y="1080" width="200" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 198px; height: 1px; padding-top: 1100px; margin-left: 522px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">zipped_list</div></div></div></foreignObject><text x="621" y="1109" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">zipped_list</text></switch></g><rect x="1869.5" y="1500" width="380" height="40" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 378px; height: 1px; padding-top: 1520px; margin-left: 1871px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 30px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><span style="font-family: &quot;courier new&quot;"><font style="font-size: 30px">zip(*zipped_list)</font></span></div></div></div></foreignObject><text x="2060" y="1529" fill="#000000" font-family="Courier New" font-size="30px" text-anchor="middle">zip(*zipped_list)</text></switch></g><path d="M 1181 1110 L 1873.13 1110" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1878.38 1110 L 1871.38 1113.5 L 1873.13 1110 L 1871.38 1106.5 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 161 190 L 489.69 613.13" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 492.91 617.28 L 485.86 613.9 L 489.69 613.13 L 491.38 609.6 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 601 180 L 638.21 609.34" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 638.66 614.57 L 634.57 607.89 L 638.21 609.34 L 641.55 607.29 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 2059.5 1300 L 2059.5 1493.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 2059.5 1498.88 L 2056 1491.88 L 2059.5 1493.63 L 2063 1491.88 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="1959.5" y="1070" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1110px; margin-left: 1961px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">2</div></div></div></foreignObject><text x="2000" y="1116" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">2</text></switch></g><rect x="1959.5" y="1190" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1230px; margin-left: 1961px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">3</div></div></div></foreignObject><text x="2000" y="1236" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">3</text></switch></g><rect x="1959.5" y="950" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 990px; margin-left: 1961px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">1</div></div></div></foreignObject><text x="2000" y="996" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">1</text></switch></g><rect x="2079.5" y="950" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 990px; margin-left: 2081px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'A'</div></div></div></foreignObject><text x="2120" y="996" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'A'</text></switch></g><rect x="2079.5" y="1070" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1110px; margin-left: 2081px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'B'</div></div></div></foreignObject><text x="2120" y="1116" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'B'</text></switch></g><rect x="2079.5" y="1190" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1230px; margin-left: 2081px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'C'</div></div></div></foreignObject><text x="2120" y="1236" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'C'</text></switch></g><rect x="1899.5" y="960" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1000px; margin-left: 1901px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="1920" y="1018" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="1899.5" y="1080" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1120px; margin-left: 1901px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="1920" y="1138" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="1899.5" y="1200" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1240px; margin-left: 1901px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="1920" y="1258" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="2039.5" y="970" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1010px; margin-left: 2041px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="2060" y="1028" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="2039.5" y="1080" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1120px; margin-left: 2041px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="2060" y="1138" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="2039.5" y="1200" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1240px; margin-left: 2041px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="2060" y="1258" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="2169.5" y="960" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1000px; margin-left: 2171px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="2190" y="1018" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="2169.5" y="1080" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1120px; margin-left: 2171px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="2190" y="1138" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="2169.5" y="1200" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1240px; margin-left: 2171px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="2190" y="1258" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="2359.5" y="1470" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1510px; margin-left: 2361px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">2</div></div></div></foreignObject><text x="2400" y="1516" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">2</text></switch></g><rect x="2359.5" y="1590" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1630px; margin-left: 2361px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">3</div></div></div></foreignObject><text x="2400" y="1636" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">3</text></switch></g><rect x="2359.5" y="1350" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1390px; margin-left: 2361px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">1</div></div></div></foreignObject><text x="2400" y="1396" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">1</text></switch></g><rect x="2479.5" y="1350" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1390px; margin-left: 2481px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'A'</div></div></div></foreignObject><text x="2520" y="1396" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'A'</text></switch></g><rect x="2479.5" y="1470" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1510px; margin-left: 2481px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'B'</div></div></div></foreignObject><text x="2520" y="1516" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'B'</text></switch></g><rect x="2479.5" y="1590" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1630px; margin-left: 2481px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'C'</div></div></div></foreignObject><text x="2520" y="1636" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'C'</text></switch></g><rect x="2299.5" y="1360" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1400px; margin-left: 2301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="2320" y="1418" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="2299.5" y="1480" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1520px; margin-left: 2301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="2320" y="1538" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="2299.5" y="1600" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1640px; margin-left: 2301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">(</div></div></div></foreignObject><text x="2320" y="1658" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">(</text></switch></g><rect x="2439.5" y="1370" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1410px; margin-left: 2441px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="2460" y="1428" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="2439.5" y="1480" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1520px; margin-left: 2441px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="2460" y="1538" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="2439.5" y="1600" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1640px; margin-left: 2441px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; "><font style="font-size: 40px">,</font></div></div></div></foreignObject><text x="2460" y="1658" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">,</text></switch></g><rect x="2569.5" y="1360" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1400px; margin-left: 2571px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="2590" y="1418" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="2569.5" y="1480" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1520px; margin-left: 2571px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="2590" y="1538" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><rect x="2569.5" y="1600" width="40" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 1640px; margin-left: 2571px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">)</div></div></div></foreignObject><text x="2590" y="1658" fill="#000000" font-family="Courier New" font-size="60px" text-anchor="middle">)</text></switch></g><path d="M 1161 510 L 1161 803.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1161 808.88 L 1157.5 801.88 L 1161 803.63 L 1164.5 801.88 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="1161" y="615" width="110" height="50" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 108px; height: 1px; padding-top: 640px; margin-left: 1162px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">zip/<br />combine</div></div></div></foreignObject><text x="1216" y="646" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">zip/...</text></switch></g><image x="1280.5" y="554.5" width="148.5" height="192.98" xlink:href="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQyH7f3uDMHUtxVLR5IHjfPg1zBNVQG1Aa1d0Cmp2Cq93ZhOim7&amp;usqp=CAU" preserveAspectRatio="none" transform="rotate(-180,1355.25,651.49)"/><path d="M 2639.5 1350 L 2639.5 1643.63" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 2639.5 1648.88 L 2636 1641.88 L 2639.5 1643.63 L 2643 1641.88 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="2639.5" y="1455" width="110" height="50" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 108px; height: 1px; padding-top: 1480px; margin-left: 2641px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">zip/<br />combine</div></div></div></foreignObject><text x="2695" y="1486" fill="#000000" font-family="Helvetica" font-size="20px" text-anchor="middle">zip/...</text></switch></g><image x="2809" y="1394.5" width="148.5" height="192.98" xlink:href="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQyH7f3uDMHUtxVLR5IHjfPg1zBNVQG1Aa1d0Cmp2Cq93ZhOim7&amp;usqp=CAU" preserveAspectRatio="none" transform="rotate(-180,2883.75,1491.49)"/><rect x="2299.5" y="1790" width="80" height="240" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><rect x="2299.5" y="1870" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1910px; margin-left: 2301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">2</div></div></div></foreignObject><text x="2340" y="1916" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">2</text></switch></g><rect x="2299.5" y="1950" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1990px; margin-left: 2301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">3</div></div></div></foreignObject><text x="2340" y="1996" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">3</text></switch></g><rect x="2299.5" y="1790" width="80" height="80" fill="#fff2cc" stroke="#d6b656" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1830px; margin-left: 2301px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">1</div></div></div></foreignObject><text x="2340" y="1836" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">1</text></switch></g><rect x="2559.5" y="1790" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1830px; margin-left: 2561px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'A'</div></div></div></foreignObject><text x="2600" y="1836" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'A'</text></switch></g><rect x="2559.5" y="1870" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1910px; margin-left: 2561px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'B'</div></div></div></foreignObject><text x="2600" y="1916" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'B'</text></switch></g><rect x="2559.5" y="1950" width="80" height="80" fill="#dae8fc" stroke="#6c8ebf" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 1990px; margin-left: 2561px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 20px; font-family: Courier New; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">'C'</div></div></div></foreignObject><text x="2600" y="1996" fill="#000000" font-family="Courier New" font-size="20px" text-anchor="middle">'C'</text></switch></g><path d="M 2399.5 1690 L 2343.03 1774.7" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 2340.12 1779.07 L 2341.09 1771.3 L 2343.03 1774.7 L 2346.92 1775.19 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><path d="M 2519.5 1690 L 2595.27 1775.24" fill="none" stroke="#000000" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 2598.76 1779.16 L 2591.49 1776.26 L 2595.27 1775.24 L 2596.72 1771.61 Z" fill="#000000" stroke="#000000" stroke-miterlimit="10" pointer-events="all"/><rect x="1" y="440" width="120" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 480px; margin-left: 2px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">zip</div></div></div></foreignObject><text x="61" y="498" fill="#000000" font-family="Helvetica" font-size="60px" text-anchor="middle">zip</text></switch></g><rect x="1581" y="860" width="180" height="80" fill="none" stroke="none" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 178px; height: 1px; padding-top: 900px; margin-left: 1582px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 60px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">unzip</div></div></div></foreignObject><text x="1671" y="918" fill="#000000" font-family="Helvetica" font-size="60px" text-anchor="middle">unzip</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://desk.draw.io/support/solutions/articles/16000042487" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Viewer does not support full SVG 1.1</text></a></switch></svg>

### Sorting in parallel

Combine two lists and sort them at the same time. 


```python
numbers = [2, 4, 3, 1]
```


```python
letters = ['b', 'a', 'd', 'c']
```


```python
data1 = list(zip(numbers, letters))
data1
```




    [(2, 'b'), (4, 'a'), (3, 'd'), (1, 'c')]




```python
data1.sort() # sort by numbers
data1
```




    [(1, 'c'), (2, 'b'), (3, 'd'), (4, 'a')]




```python
data2 = list(zip(letters, numbers))
data2
```




    [('b', 2), ('a', 4), ('d', 3), ('c', 1)]




```python
data2.sort() # sort by letters
data2
```




    [('a', 4), ('b', 2), ('c', 1), ('d', 3)]



Use `sorted()` and `zip()` together to achieve a similar result


```python
data = sorted(zip(letters, numbers))
data
```




    [('a', 4), ('b', 2), ('c', 1), ('d', 3)]



### Calculating in pairs


```python
total_sales = [52000.00, 51000.00, 48000.00]
prod_cost = [46800.00, 45900.00, 43200.00]

for sales, costs in zip(total_sales, prod_cost):
    profit = sales - costs
    print(f'Profit: {profit}')
```

    Profit: 5200.0
    Profit: 5100.0
    Profit: 4800.0


### Building Dictionaries


```python
fields = ['name', 'last_name', 'age', 'job']
values = ['John', 'Doe', '45', 'Python Developer']

a_dict = dict(zip(fields, values))
```


```python
a_dict
```




    {'name': 'John', 'last_name': 'Doe', 'age': '45', 'job': 'Python Developer'}



Update an existing dictionary by combining `zip()` with `dict.update()`.


```python
new_job = ['Python Consultant']
field = ['job']

a_dict.update(zip(field, new_job))
a_dict
```




    {'name': 'John', 'last_name': 'Doe', 'age': '45', 'job': 'Python Consultant'}


