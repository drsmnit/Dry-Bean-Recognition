<p style="text-align: center;"><strong><span style="font-size: 22px;">Dry Bean Dataset</span></strong></p>
<p><br></p>
<p><strong><u>Data Set information</u></strong></p>
<ol>
    <li><strong>Abstract</strong>: Images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. A total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.</li>
    <li><strong>Data Type</strong>: Multivariate</li>
    <li><strong>Task</strong>: Classification</li>
    <li><strong>Attribute Type</strong>: Categorical, Integer, Real</li>
    <li><strong>Area</strong>: CS / Engineering</li>
    <li><strong>Format Type</strong>: Matrix</li>
    <li><strong>Number of Instances (records in your data set)</strong>: 13611</li>
    <li><strong>Number of Attributes (fields within each record)</strong>: &nbsp;17</li>
</ol>
<p><strong>Relevant Information</strong>: Seven different types of dry beans were used in this research, taking into account the features such as form, shape, type, and structure by the market situation. A computer vision system was developed to distinguish seven different registered varieties of dry beans with similar features in order to obtain uniform seed classification. For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.</p>
<p><strong>Attribute Information</strong>:</p>
<ol>
    <li>Area (A): The area of a bean zone and the number of pixels within its boundaries.</li>
    <li>Perimeter (P): Bean circumference is defined as the length of its border.</li>
    <li>Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.</li>
    <li>Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.</li>
    <li>Aspect ratio (K): Defines the relationship between L and l.</li>
    <li>Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.</li>
    <li>Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.</li>
    <li>Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.</li>
    <li>Extent (Ex): The ratio of the pixels in the bounding box to the bean area.</li>
    <li>Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.</li>
    <li>Roundness (R): Calculated with the following formula: (4piA)/(P^2)</li>
    <li>Compactness (CO): Measures the roundness of an object: Ed/L</li>
    <li>ShapeFactor1 (SF1)</li>
    <li>ShapeFactor2 (SF2)</li>
    <li>ShapeFactor3 (SF3)</li>
    <li>ShapeFactor4 (SF4)</li>
    <li>Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)</li>
</ol>

<strong>Import Libraries/Dataset</strong>
