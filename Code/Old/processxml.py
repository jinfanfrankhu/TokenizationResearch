import csv
import xml.etree.ElementTree as ET

def xml_to_csv(xml_file, csv_file):
    # Define namespace if necessary
    namespace = {'mw': "http://www.mediawiki.org/xml/export-0.11/"}

    # Open CSV file for writing
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header row
        writer.writerow(["Title", "Namespace", "Page ID", "Revision ID", "Timestamp", "Contributor Username", "Contributor User ID", "Text"])

        # Stream parse the XML
        context = ET.iterparse(xml_file, events=("start", "end"))
        context = iter(context)
        event, root = next(context)  # Get the root element

        for event, elem in context:
            if event == "end" and elem.tag.endswith("page"):
                title = elem.find('mw:title', namespace).text
                ns = elem.find('mw:ns', namespace).text
                page_id = elem.find('mw:id', namespace).text

                # Process revisions
                for revision in elem.findall('mw:revision', namespace):
                    revision_id = revision.find('mw:id', namespace).text
                    timestamp = revision.find('mw:timestamp', namespace).text

                    contributor = revision.find('mw:contributor', namespace)
                    username = contributor.find('mw:username', namespace).text if contributor.find('mw:username', namespace) else None
                    user_id = contributor.find('mw:id', namespace).text if contributor.find('mw:id', namespace) else None

                    text = revision.find('mw:text', namespace).text

                    # Write row to CSV
                    writer.writerow([title, ns, page_id, revision_id, timestamp, username, user_id, text])

                root.clear()  # Free memory

# Example usage
xml_file = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\trwiki-20241201-pages-meta-current.xml'
csv_file = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\output.csv'
xml_to_csv(xml_file, csv_file)
print(f"XML data has been written to {csv_file}")
