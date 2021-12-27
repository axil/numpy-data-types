NumPy Data Types
################

.. image:: img/numpy-data-types/numpy_types_diagram.png
  :alt: NumPy Types Diagram

NumPy, one of the most popular Python libraries for both data science and scientific computing, is pretty omnivorous when it comes to data types: it can handle just everything.

It has its own set of ‘native’ types which it is capable of processing at full speed but it can also work with pretty much anything known to Python.

The article consists of 7 parts:

1. Integers
2. Floats (including Fractions and Decimals)
3. Bools
4. Strings
5. Datetimes
6. Combinations thereof
7. Type Checks

***********
1. Integers
***********

The integer types table in NumPy is absolutely trivial for anyone with minimal experience in C/C++: 

.. image:: img/numpy-data-types/integers.png
  :alt: NumPy Integer Types

Just like in C/C++, `u` stands for 'unsigned' and the number is the amount of bits used to store the variable in memory (eg int64 is a 8-bytes-wide signed integer).

When you feed a Python int into NumPy, it gets converted into a native NumPy type called np.int32 (or np.int64 depending on the OS, Python version and the magnitude of the initializers):

.. code:: python

        >>> np.array([1, 2, 3]).dtype      
        dtype('int32')                   # int32 on windows, int64 on linux and macos

If you’re unhappy with the 'flavor' of the integer type that NumPy has chosen for you, you can specify one explicitly: np.array([1,2,3], np.uint8) or np.array([1,2,3], 'uint8').

NumPy works best when the width of the array elements is fixed. It is faster and takes less memory, but unlike an ordinary Python int (that works in arbitrary precision arithmetic) the value will wrap when it crosses the maximum (or minimum) value for the corresponding data type:

.. image:: img/numpy-data-types/int_wrapping.png
  :alt: Int Wrapping

.. code:: python

        >>> np.array([255], np.uint8) + 1 # 2**8-1 is INT_MAX for uint8
        array([0], dtype=uint8)

        >>> np.array([2**31-1])           # 2**31-1 is INT_MAX for int32
        array([2147483647]) 

        >>> np.array([2**31–1]) + 1       # or np.array([2**31-1], np.int32)+1 on linux
        array([-2147483648]) 

        >>> np.array([2**63-1]) + 1       # always np.int64 since v > 2**32-1
        array([-9223372036854775808])

\— not even a warning here!

With scalars it is a different story: first NumPy tries it best to promote the value to a wider type, then, if there is none, fires the overflow warning (to avoid flooding the output with warnings—only once):

.. code:: python

        >>> np.array([255], np.uint8)[0] + 1   # ok, promoted to int32(win)/int64(linux)
        256                                     
        >>> np.array([2**31-1])[0] + 1         # warning!
        RuntimeWarning: overflow encountered in long_scalars
        -2147483648
        >>> np.array([2**63-1])[0] + 1         # ok, warned already
        -9223372036854775808

The reasoning behind such a discrimination is like this:

    Unlike true floating point errors (where the hardware FPU sets a flag whenever it does an atomic operation that overflows), we need to implement the integer overflow detection ourselves. We do it on the scalars, but not arrays because it would be too slow to implement for every atomic operation on arrays. *Robert Kern, one of the NumPy core developers*

You can turn it into an error:

.. code:: python

        >>> with np.errstate(over='raise'):
        >>>    print(np.array([2**31-1])[0]+1)
        FloatingPointError: overflow encountered in long_scalars

(although the name FloatingPointError for an *integer* overflow looks a bit misleading.)

... or suppress it entirely:

.. code:: python

        >>> with np.errstate(over='ignore'):
        >>>    print(np.array([2**31-1])[0]+1)
        -2147483648

But you can’t expect it to be detected when dealing with arrays (even with the 0-dimensional ones!).

NumPy also has a bunch of C-style aliases (eg. np.byte np.int8, np.short=np.int16, np.intc=int whichever width it has in C etc), but they are getting gradually phased out (eg `deprecation of np.long in NumPy v1.20.0 <https://numpy.org/devdocs/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated>`_) as 'explicit is better than implicit' (but see a present-day usage of np.longdouble below). 

And yet some more exotic aliases: 

* `np.int_` is np.int32 on 64bit windows but int64 on 64bit linux, used to designate the 'default' int. Specifying `np.int_` as a dtype is the same thing as specifying int and means "do what you would do if I didn't specify any dtype at all": np.array([1,2,3]), np.array([1,2,3], `np.int_`) and np.array([1,2,3], int) are all the same thing.

* `np.intp` is np.int32 on 32bit python but np.int64 on 64bit python, ≈ssize_t in C, used in cython

Finally, if for some reason you need arbitrary-precision integers (Python ints) in ndarrays, NumPy is capable of doing that, too:

.. code:: python

        >>> a = np.array([10], dtype=object)
        >>> len(str(a**1000))                   # '[1000...0]'
        1003

— but without the usual speedup as it will have to store references instead of the numbers themselves, keep boxing/unboxing Python objects when processing, etc.

*********
2. Floats
*********

As Python did not diverge from IEEE 754-standardized C double type, the floating type transition from Python to NumPy is pretty much hassle-free:

.. image:: img/numpy-data-types/floats.png
  :alt: NumPy Floating Types

\* *This is the number reported by np.finfo(np.float<nn>).precision. As usual with floats, depending on what you mean by significant digits it may be* |br|
|_| |_| |_| *– 15* (`FLT_DIG <https://en.cppreference.com/w/cpp/types/numeric_limits/digits10>`_) *or 17* ( `FLT_DECIMAL_DIG <https://en.cppreference.com/w/cpp/types/numeric_limits/max_digits10>`_) *for np.float64,* |br|
|_| |_| |_| *– 6 (FLT_DIG) or 9 (FLT_DECIMAL_DIG) for np.float32, etc.*

** *Support for np.float128 is somewhat limited: it is unix-only (=linux and mac; not available on windows). Also the names float96/float128 are highly misleading. Under the hood it is not __float128 but whichever longdouble means in the local C++ flavor. On x86_64 linux it is float80 (padded with zeros for memory alignment) which is certainly wider than float64, but comes at the cost of slower processing speed. Also you risk losing precision if you inadvertently convert it to Python float type (which is 64bit wide). For better portability it is recommended to use an alias np.longdouble instead of np.float96 / np.float128 because that’s what will be used internally anyway.*

Just like in pure Python, NumPy floats exactly represent integers—but only below a certain level (limited by the number of the significant digits):

.. code:: python

        >>> a = np.array([2**24], np.float32); a    # 2^(mantissa_bits+1)
        array([16777216.], dtype=float32)
        >>> a+1
        array([16777216.], dtype=float32)       
        >>> 9279945539648888.0+1    # for float64 it is 2.**53
        9279945539648888.0               
        >>> len('9279945539648888') # Don't trust the 16th decimal digit!
        16

Also exactly representable are fractions like 0.5, 0.125, 0.875 where the denominator is a power of 2 (0.5=1/2, 0.125=1/8, 0.875 =7/8, etc).

Any other denominator will result in a rounding error so that 0.1+0.2!=0.3. The standard approach of dealing with this problem is to compare them with a relative tolerance (to compare two non-zero arguments) and absolute tolerance (if one of the arguments is zero). For scalars it is handled by `math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)`, for NumPy arrays there’s a vectorized version `np.isclose(a, b, rtol=1e-05, atol=1e-08)`. Note that the tolerance arguments have different names and defaults.

For the financial data decimal.Decimal type is handy as it involves no tolerances at all:

.. code:: python

        >>> from decimal import Decimal as D
        >>> a = np.array([D('0.1'), D('0.2')]); a
        array([Decimal('0.1'), Decimal('0.2')], dtype=object)
        >>> a.sum()                     # == Decimal('0.3'), exactly      
        Decimal('0.3')

But Decimal type is not a silver bullet: it also has rounding errors. The only problem it solves is the exact representation of decimal fractions that humans are so used to. Plus it doesn’t support anything more complicated than arithmetic operations and a square root and runs slower than floats.

For pure mathematic calculations fractions.Fraction can be used:

.. code:: python

        >>> from fractions import Fraction
        >>> a = np.array([1, 2]) + Fraction(); a
        array([Fraction(1, 1), Fraction(2, 1)], dtype=object)
        >>> a/=10; a
        array([Fraction(1, 10), Fraction(1, 5)], dtype=object)
        >>> a.sum()
        Fraction(3, 10)

It can represent any rational number, but pi and exp are out of luck )

Both Decimal and Fraction are not native types for NumPy but it is capable of working with them with all the niceties like multi-dimensions and fancy indexing, albeight at the cost of slower processing speed than that of native ints or floats.

Complex numbers are processed no differently than floats with extra convenience functions with intuitive names like np.real(z), np.imag(z), np.abs(z), np.angle(z) that work on both scalars and arrays as a whole.

More insights on floats can be found in the following sources:

.. |_| unicode:: 0xA0 
   :trim:

.. |br| raw:: html

  <br/>

\• short and nicely illustrated `‘Half-Precision Floating-Point, Visualized’ <https://observablehq.com/@rreusser/half-precision-floating-point-visualized>`_ [2] |br|
|_| |_| |_| — eg What’s the difference between normal and subnormal numbers?

\• more lengthy but very to-the-point, a dedicated website `‘Floating point guide’ <https://floating-point-gui.de/>`_ [3]  |br|
|_| |_| |_| — eg Why 0.1+0.2!=0.3? 

\• long-read, a deep and thorough `What Every Computer Scientist Should Know About Floating-Point Arithmetic, Appendix D <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>`_ [4]  |br|
|_| |_| |_| — eg What’s the difference between catastrophic vs benign cancellation?

********
3. Bools
********

The boolean values are stored as single bytes for better performance. `np.bool_` is a separate type from Python’s bool because it doesn’t need reference counting and a link to the baseclass required for any pure Python type. So if you think that using 8 bits to store one bit of information is excessive look at this:

.. code:: python

        >>> sys.getsizeof(True)
        28

np.bool is 28 times more memory efficient than Python’s bool ) – though in real-world scenarios the rate is lower: when you pack NumPy bools into an array, they will take 1 byte each, but if you pack Python bools into a list it will reference the same two values every time, costing effectively 8 bytes per element on x86_64:

.. image:: img/numpy-data-types/bools.png
  :alt: NumPy Boolean Type


The underlines in `bool_`, `int_`, etc are there to avoid clashes with Python’s types. It’s a bad idea to use reserved keywords for other things, but in this case it has an additional advantage of allowing (a generally discouraged, but useful in rare cases) from NumPy import * without shadowing Python bools, ints, etc. As of today, np.bool still works but displays a deprecation warning.

**********
4. Strings
**********

Initializing a NumPy array with a list of Python strings packs them into a fixed-width native NumPy dtype called `np.str_`. Reserving a space necessary to fit the longest string for every element might look wasteful (especially in the fixed USC-4 encoding as opposed to ‘dynamic’ choice of the UTF width in Python str)

.. code:: python

        >>> np.array(['abcde', 'x', 'y', 'z'])        # 4 bytes per any character
        array(['abcde', 'x', 'y', 'z'], dtype='<U5')  # => 5*4 bytes per element

The abbreviation ‘<U4’ comes from the so called array protocol introduced in 2005. It means ‘little-endian USC-4-encoded string, 5 elements long’ (USC-4≈UTF-32, a fixed width, 4-bytes per character encoding). Every NumPy type has an abbreviation as unreadable as this one, luckily have they adopted human-readable names at least for the most used dtypes.

Another option is to keep references to Python strs in a NumPy array of objects:

.. code:: python

        >>> np.array(['abcde', 'x', 'y', 'z'], object)     # 1 byte per ascii character
        array(['abcde', 'x', 'y', 'z'], dtype=object)      # => 49+len(el) per element

The first array memory footprint amounts to 164 bytes, the second one takes 128 bytes for the array itself + 154 bytes for the three python strs:

.. image:: img/numpy-data-types/str.png
  :alt: NumPy Str_ Type

Depending on the relative lengths of the strings and the number of the repeated string either one approach can be a significant win or the other.

If you're dealing with a raw sequence of bytes NumPy has a fixed-length version of a Python bytes type called `np.bytes_`:

.. code:: python

        >>> np.array([b'abcde', b'x', b'y', b'z'])    # 1 byte per ascii character
        array([b'abcde', b'x', b'y', b'z'], dtype='|S5') # => 5 bytes per element

Here `|S5` means ‘endianness-unappliable sequence of bytes 5 elements long’.

Once again, an alternative is to store the Python `bytes` in the NumPy array of objects. 

.. code:: python

        >>> np.array([b'abcde', b'x', b'z'], object)  # 1 byte per ascii character
        array([b'abcde', b'x', b'z'], dtype=object)   # => 33+len(el) per element

This time the first array takes 124 bytes, the second one is the same 128 bytes for the array itself + 106 bytes for the three python `bytes`:

.. image:: img/numpy-data-types/bytes.png
  :alt: NumPy Bytes_ Type

We see that `str_` is smaller again, yet for more diverse lengths str can take the win.

As for the native `np.str_` and `np.bytes_` types, NumPy has a handful of common string operations mirroring str methods living in the np.char module that operate over the whole array:

.. code:: python

        >>> np.char.upper(np.array([['a','b'],['c','d']]))
        array([['A', 'B'],
        ['C', 'D']], dtype='<U1')

With object-mode strings the loops must happen on the Python level:

.. code:: python

        >>> np.vectorize(lambda x: x.upper(), otypes=[object])(a)
        array([['A', 'B'],
            ['C', 'D']], dtype=object)

According to my benchmarks, basic operations work somewhat faster with str than with `np.str_`.

****************
5. Datetimes
****************

An interesting data type, capable of counting time with configurable granularity — from years to attoseconds (an aspect in which other datetime libs tend to rely on the underlying OS) — represented invariably by int64.

Years granularity means ‘just count the years’ — no real improvement against storing years as an integer. Days granularity is the equivalent of Python’s datetime.date. Microseconds (or nanoseconds depending on the OS) is the equivalent of Python’s datetime.datetime. And everything below is unique to np.datetime64.

When creating an array you choose if you are ok with the default microseconds or you insist on nanoseconds or what not and it’ll give you 2⁶³ equidistant moments measured in the corresponding units of time to either side of 1 Jan 1970.

.. code:: python

        >>> np.array([dt.utcnow()], dtype=np.datetime64)
        array(['2021-12-24T18:14:00.403438'], dtype='datetime64[us]')

One downside of it is that all the times are naive: they know nothing of daylight saving and are not capable of being converted from one timezone to another. So it is not a replacement for pytz, rather a complement to it.


***********************
6. Combinations thereof
***********************

A structured dtype allows to create a custom type using the types described above as the basic building blocks. Typical example is an RGB pixel: a 4 bytes long type, in which the colors can be accessed by name: 

.. code:: python

        >>> rgb = np.dtype([('x', np.uint8), ('y', np.uint8), ('z', np.uint8)])
        >>> a = np.zeros(5, z); a
        array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
            dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
        >>> a[0]
        (0, 0, 0)
        >>> a[0]['x']
        0
        >>> a[0]['x'] = 10
        >>> a
        array([(10, 0, 0), ( 0, 0, 0), ( 0, 0, 0), ( 0, 0, 0), ( 0, 0, 0)],
            dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
        >>> a['z'] = 5
        >>> a
        array([(10, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5)],
            dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])

To be able to access the fields as attributes, a recarray can be used:

.. code:: python

        >>> b = a.view(np.recarray)
        >>> b
        rec.array([(10, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5)],
                  dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
        >>> b[0].x
        10
        >>> b.y=7; b
        rec.array([(10, 7, 5), ( 0, 7, 5), ( 0, 7, 5), ( 0, 7, 5), ( 0, 7, 5)],
          dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
        
Sure enough, recarray can be created on its own, without being a view of something else.
Types for structured dtypes do not necessarily need to be homogenic and can even
include subarrays.

**************
7. Type Checks
**************

One way to check NumPy array type is to run isinstance against its element:

.. code:: python

        >>> a = np.array([1, 2, 3])
        >>> v = a[0]
        >>> isinstance(v, np.int32)    # might be np.int64 on a different OS
        True

All the NumPy types are interconnected in an inheritance tree displayed in the top of the article (blue=abstract classes, green=numeric types, yellow=others) so instead of specifying a whole list of types like isinstance(v, [np.int32, np.int64, etc]) you can write more compact typechecks like

.. code:: python

        >>> isinstance(v, np.integer)        # true for all integers
        True
        >>> isinstance(v, np.number)         # true for integers and floats
        True
        >>> isinstance(v, np.floating)       # true for floats except complex
        False
        >>> isinstance(v, np.complexfloating) # true for complex floats only 
        False

The downside of this method is that it only works against a value of the array, not against the array itself. Which is not useful when the array is empty, for example. Checking the type of the array is more tricky.

For basic types the == operator does the job for a single type check:

.. code:: python

        >>> a.dtype == np.int32
        True
        >>> a.dtype == np.int64
        False

and in operator for checking against a group of types:

.. code:: python

        >>> x.dtype in (np.half, np.single, np.double, np.longdouble)
        False

But for more sophisticated types like `np.str_` or `np.datetime64` it doesn’t.

The recommended way [5] of checking the dtype against the abstract types is

.. code:: python

        >>> np.issubdtype(a.dtype, np.integer)
        True
        >>> np.issubdtype(a.dtype, np.floating)
        False

It works with all native NumPy types, but the necessity of this method looks somewhat non-obvious: what’s wrong with good oldisinstance? Obviously the complexity of dtypes inheritance structure (they are constructed ‘on the fly’!) didn’t allow to do it according to principle of the least astonishment.

Yet another method is to use (undocumented, but used in SciPy/NumPy code bases) np.typecodes dictionary. The tree it represents is way less branchy:

.. code:: python

        >>> np.typecodes
        {'Character': 'c',
        'Integer': 'bhilqp',
        'UnsignedInteger': 'BHILQP',
        'Float': 'efdg',
        'Complex': 'FDG',
        'AllInteger': 'bBhHiIlLqQpP',
        'AllFloat': 'efdgFDG',
        'Datetime': 'Mm',
        'All': '?bhilqpBHILQPefdgFDGSUVOMm'}

And the usage is like

.. code:: python

        >>> a.dtype.kind in np.typecodes['AllInteger']
        True
        >>> a.dtype.kind in np.typecodes['Datetime']
        False

This approach looks more hackish yet less magical than issubdtype.

References

1. Ricky Reusser, `Half-Precision Floating-Point, Visualized <https://observablehq.com/@rreusser/half-precision-floating-point-visualized>`_

2. Floating point guide https://floating-point-gui.de/

3. David Goldberg, `What Every Computer Scientist Should Know About Floating-Point Arithmetic, Appendix D <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>`_

4. NumPy issue `#17325 <https://github.com/numpy/numpy/issues/17325>`_, Add a canonical way to determine if dtype is integer, floating point or complex

