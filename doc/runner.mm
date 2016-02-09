<map version="docear 1.1" project="152C1538D029ED0QM6VNNR0P0P3TNAL6ISAY" project_last_home="file:/home/perrette/Docear/projects/runner" dcr_id="1454942424390_ajuh8bmkz8r5u4mo8qaius03v">
<!--To view this file, download Docear - The Academic Literature Suite from http://www.docear.org -->
<attribute_registry FONT_SIZE="6">
    <attribute_name MANUAL="true" NAME="setup_outdir">
        <attribute_value VALUE=""/>
    </attribute_name>
</attribute_registry>
<node TEXT="runner" FOLDED="false" ID="ID_1723255651" CREATED="1283093380553" MODIFIED="1454944650971"><hook NAME="MapStyle">
    <properties show_note_icons="false" show_icon_for_attributes="true" show_notes_in_map="true"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node">
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="right">
<stylenode LOCALIZED_TEXT="default" MAX_WIDTH="600" COLOR="#000000" STYLE="as_parent">
<font NAME="SansSerif" SIZE="10" BOLD="false" ITALIC="false"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.note"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="right">
<stylenode LOCALIZED_TEXT="styles.topic" COLOR="#18898b" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subtopic" COLOR="#cc3300" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subsubtopic" COLOR="#669900">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.important">
<icon BUILTIN="yes"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="right">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#000000">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#0033ff">
<font SIZE="16"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#00b439">
<font SIZE="14"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#990000">
<font SIZE="12"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#111111">
<font SIZE="10"/>
</stylenode>
</stylenode>
</stylenode>
</map_styles>
</hook>
<hook NAME="AutomaticEdgeColor" COUNTER="1"/>
<attribute_layout NAME_WIDTH="25" VALUE_WIDTH="25"/>
<node TEXT="disk" POSITION="right" ID="ID_1611886961" CREATED="1454942661616" MODIFIED="1454947787028" HGAP="10" VSHIFT="-160" MOVED="1454946220657">
<edge COLOR="#ff00ff"/>
<hook NAME="FreeNode"/>
<node TEXT="output dir" ID="ID_960442987" CREATED="1454945545597" MODIFIED="1454945631420" VSHIFT="-20">
<node TEXT="modified params&apos; file" ID="ID_950537665" CREATED="1454942708696" MODIFIED="1454946565585" HGAP="10" VSHIFT="-40"/>
</node>
<node TEXT="input dir" ID="ID_227570187" CREATED="1454945593677" MODIFIED="1454947563317">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_743386548" STARTINCLINATION="-14;44;" ENDINCLINATION="-101;-33;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<node TEXT="executable" ID="ID_1929890488" CREATED="1454942701344" MODIFIED="1454945725675" HGAP="50" VSHIFT="10" MOVED="1454942841255"/>
<node TEXT="default params&apos; file" ID="ID_1912534819" CREATED="1454945696508" MODIFIED="1454945727436" HGAP="40" VSHIFT="20" MOVED="1454945879703"/>
</node>
</node>
<node TEXT="toolbox" POSITION="right" ID="ID_1835458903" CREATED="1454944410449" MODIFIED="1454948030567" HGAP="-400" VSHIFT="-80" MOVED="1454946220660">
<edge COLOR="#7c0000"/>
<hook NAME="FreeNode"/>
<richcontent TYPE="NOTE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      - system : run executable, submit job etc...
    </p>
    <p>
      - probabilistic : factorial combination of parameters, etc...
    </p>
  </body>
</html>

</richcontent>
</node>
<node TEXT="Model" POSITION="right" ID="ID_743386548" CREATED="1454942432465" MODIFIED="1454947883284" HGAP="70" VSHIFT="-10" MOVED="1454946112659">
<edge COLOR="#ff0000"/>
<attribute_layout NAME_WIDTH="90" VALUE_WIDTH="114"/>
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_960442987" STARTINCLINATION="33;-30;" ENDINCLINATION="194;-23;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="NOTE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      - setup_outdir : setup output directory (e.g. write parameter file)
    </p>
    <p>
      - run executable in terminal or submit Job
    </p>
  </body>
</html>

</richcontent>
<node TEXT="Params" ID="ID_1623254892" CREATED="1454942460409" MODIFIED="1454947954515" HGAP="-265" VSHIFT="120" MOVED="1454946239896">
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_950537665" STARTINCLINATION="127;-54;" ENDINCLINATION="246;0;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="NOTE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      List of model parameters (useful for I/O).
    </p>
    <p>
      Each param has a number of attributes:
    </p>
    <p>
      - name
    </p>
    <p>
      - value
    </p>
    <p>
      - units
    </p>
    <p>
      - etc ...
    </p>
  </body>
</html>

</richcontent>
</node>
</node>
<node TEXT="command-line interface" POSITION="left" ID="ID_1475036761" CREATED="1454942962430" MODIFIED="1454947493034" HGAP="100" VSHIFT="70" MOVED="1454947452918">
<edge COLOR="#ff0000"/>
<richcontent TYPE="NOTE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      - Initialize model via input parameter file.
    </p>
    <p>
      - Initialize ensemble via paramters to modify
    </p>
  </body>
</html>

</richcontent>
</node>
<node TEXT="Ensemble" POSITION="left" ID="ID_1362713318" CREATED="1454942523160" MODIFIED="1454948016147" HGAP="-230" VSHIFT="220" MOVED="1454947474375">
<edge COLOR="#00ff00"/>
<arrowlink SHAPE="CUBIC_CURVE" COLOR="#000000" WIDTH="2" TRANSPARENCY="80" FONT_SIZE="9" FONT_FAMILY="SansSerif" DESTINATION="ID_1623254892" STARTINCLINATION="6;-39;" ENDINCLINATION="-67;2;" STARTARROW="NONE" ENDARROW="DEFAULT"/>
<richcontent TYPE="NOTE">

<html>
  <head>
    
  </head>
  <body>
    <p>
      - list of parameter names (which can each match one parameter in Params)
    </p>
    <p>
      - list of list of parameter values (number of samples x number of params).
    </p>
  </body>
</html>

</richcontent>
<hook NAME="FreeNode"/>
</node>
</node>
</map>
